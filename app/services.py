"""分析编排：后台串行重算、重启恢复、修订交错保护。"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from . import database as db
from .analysis import compute_aalen_johansen

logger = logging.getLogger("cohort.analysis_service")

# 后台计算线程的全局控制
_worker_lock = threading.Lock()
_worker_thread: Optional[threading.Thread] = None
_running_revision: Optional[int] = None
_stop_event = threading.Event()
# 唤醒代数：ensure_worker 发现线程仍存活时递增；工作线程在"队列已空"的退场判定
# 中于锁内比对代数，从而不会漏掉恰在退场窗口内受理的新修订。
_wake_generation = 0


def _run_revision(revision: int) -> None:
    """执行单个修订的重算并落库。"""
    global _running_revision
    _running_revision = revision
    try:
        row = db.get_analysis_row(revision)
        if row is None:
            logger.warning("修订 %s 无分析记录，跳过", revision)
            return
        if row["status"] != "PENDING":
            logger.info("修订 %s 状态为 %s，无需重算", revision, row["status"])
            return
        # 遗留的 PENDING 若已不是 CURRENT（例如进程在受理事务后、旧分析作废前
        # 崩溃），直接作废，不计算也不发布。
        if not db.is_current_revision(revision):
            logger.warning("修订 %s 已非现行版本，跳过计算并置为 SUPERSEDED", revision)
            db.mark_analysis_superseded(revision)
            return

        logger.info("开始重算修订 %s", revision)
        records = db.load_records(revision)
        result = compute_aalen_johansen(records)
        # 落库内部会再次校验 CURRENT 身份；被取代则转为 SUPERSEDED 而不发布
        db.save_analysis_result(revision, result)
        after = db.get_analysis_row(revision)
        logger.info("修订 %s 重算结束，最终状态 %s", revision, after["status"])
    except Exception as exc:  # noqa: BLE001 - 后台线程需吞掉异常并落 FAILED
        logger.exception("修订 %s 重算失败", revision)
        try:
            db.mark_analysis_failed(revision, f"{type(exc).__name__}: {exc}")
        except Exception:  # pragma: no cover - 落库失败仅记录日志
            logger.exception("修订 %s 失败状态落库也失败了", revision)
    finally:
        _running_revision = None


def _worker_loop() -> None:
    """后台守护线程循环：只要还有 PENDING 修订就逐个重算。

    退场握手（修复修订入队与线程退出的交错）：
    - 每轮先记下当前唤醒代数，再查 PENDING 队列；若队列为空，则必须在
      ``_worker_lock`` 内重新检查"是否有新的唤醒代数 / 队列是否仍为空"。
    - ``ensure_worker`` 在发现线程仍存活时递增代数，因此上传受理与退场判定
      互斥：要么上传让线程看到新代数继续跑，要么线程先清空句柄、上传随后
      拉起新线程。两种顺序都不会让新的 PENDING 无人处理。
    """
    global _worker_thread
    logger.info("分析工作线程启动")
    try:
        while True:
            with _worker_lock:
                generation = _wake_generation
                if _stop_event.is_set():
                    break
            try:
                pending = db.pending_revisions()
                if pending:
                    _run_revision(pending[0])
                    continue

                # 队列暂时为空：在锁内做最终退场判定，避免与 ensure_worker 竞态
                with _worker_lock:
                    if _stop_event.is_set() or _wake_generation != generation:
                        continue  # 退场窗口内有新任务被受理，重新检查队列
                    if db.pending_revisions():
                        continue  # 受理事务已提交但唤醒尚未发生，保险起见再查一次
                    _worker_thread = None
                    break
            except Exception:  # noqa: BLE001 - 单次读取/重算异常不应杀死工作线程
                logger.exception("分析工作线程循环异常，稍后重试")
                _stop_event.wait(timeout=1.0)
    finally:
        # 无论正常退出还是异常退出，都清理自己的句柄（仅当句柄仍指向自己），
        # 让下一次 ensure_worker 能可靠地拉起新线程，而不是看到"活着的僵尸"。
        with _worker_lock:
            if _worker_thread is threading.current_thread():
                _worker_thread = None
        logger.info("分析工作线程退出")


def ensure_worker() -> bool:
    """确保有一个后台工作线程在处理 PENDING 修订。返回是否新启动了线程。"""
    global _worker_thread, _wake_generation
    with _worker_lock:
        if _worker_thread is not None:
            if _worker_thread.is_alive():
                # 线程可能正处在"已看到空队列、尚未退出"的窗口：递增代数使其
                # 在锁内重查队列，而不是直接退场把新任务晾在 PENDING。
                _wake_generation += 1
                return False
            # 线程已结束但句柄尚未被自身清理（异常终止等）：视为可重启
            _worker_thread = None
        _stop_event.clear()
        _wake_generation += 1
        t = threading.Thread(target=_worker_loop, name="analysis-worker", daemon=True)
        _worker_thread = t
        t.start()
        return True


def recompute_sync(revision: int) -> None:
    """同步重算（供测试与启动恢复使用）。"""
    db.mark_analysis_pending(revision)
    _run_revision(revision)


def current_running_revision() -> Optional[int]:
    return _running_revision


def recover_on_startup() -> None:
    """服务启动时恢复：

    - 任何遗留的非终态分析都保持/重置为 PENDING（重算失败或重启后仍待重算）；
    - 启动后台线程逐个重算，期间查询只会看到 PENDING 而不会返回旧结论。
    """
    pending = db.pending_revisions()
    if pending:
        logger.info("启动恢复：待重算修订 %s", pending)
        ensure_worker()
