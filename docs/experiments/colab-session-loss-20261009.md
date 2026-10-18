# Colab session loss — October 9, 2026

The user-launched A100 run `knee-coat384` lost its remote assignment before completing fold0. Epochs0–3 were logged; the last completed epoch at21:48:28UTC had descriptive report AUC0.8882. No completed run receipt or checkpoint was copied to the local results directory. This is an interrupted experiment, not a completed model or score comparison.

## Evidence and response

Keepalive history shows connection/transport errors from21:52 through21:57, then HTTP404 at22:20:02 and22:21:02. After two consecutive4xx responses the keepalive process stopped. At22:30 the CLI found no active backend assignment and pruned its stale local record. That operation only removed local state and stopped keepalive; it did not request remote termination. The cause of assignment loss is unknown. The evidence does not establish quota exhaustion, OOM, user cancellation or network loss as the cause.

The local driver remained waiting. Installed `jupyter_kernel_client/wsclient.py` checks connection state before waiting for a message, while its connection-close handler clears connection state without waking that waiter. The driver supplied a22-hour execution timeout, so a disconnected command can remain blocked. Parent confirmed the orphan PID5248 belonged to this exact command, preserved logs and sent SIGTERM to that local process only. Driver3918 then attempted recovery, received “session not found”, and exited143. No remote stop, new session or credential transfer occurred.

The driver's unconditional `--keep` footer says the session is still running; this is false in this failure case. Backend absence and the recovery error take precedence. Processes3918/5248/5250 are gone. Reusing the session name cannot restore the deleted assignment, and no checkpoint recovery is currently evidenced.

## Before the next Colab attempt

Fable owns the driver and its next changes. Add bounded transport/assignment-health detection so a missing runtime ends a stale local wait; this is not a training-duration limit. Make the exit footer reflect verified session state. Persist full resumable training state and logs outside the ephemeral VM at epoch boundaries through an authorized path, and verify retrieval. The current training script's best-model weights alone would not provide exact optimizer/scheduler resume; full resume state is currently written only on its time-budget exit path.

Do not claim an exact continuation from epoch3 without recovered matching state. Preserve this failure separately from any new attempt and reconcile active assignments before launching. Existing Kaggle round1fold4 and memory-revised CoAtNet224fold4 remain running independently.

Safe aggregate history, preserved local logs and process/result receipt: `outputs/analysis/reconcile-20261009-2230/`. Full CLI history and credentials were not copied into the repository. No runtime code changed in this incident response; source inspection, safe history fields, backend status, process identity and post-stop checks supplied verification.
