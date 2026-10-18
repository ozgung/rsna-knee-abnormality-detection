#!/usr/bin/env bash
# Driver (runs on the Mac) for the Track A fast-path training package on a Colab A100.
#
#   scripts/colab_tracka.sh --session knee-fold4 --fold-a 4 [--fold-b K|""] [--epochs N]
#                           [--arch TIMM_NAME] [--res PX] [--k N] [--k-eval N] [--bs N] [--labels-file NAME.parquet]
#                           [--grad-ckpt] [--win-chunk N] [--accum N] [--time-budget-h H] [--exec-timeout-h H]
#                           [--reuse-session] [--smoke] [--keep|--no-stop] [--dry-run]
#                           [--persist-dataset OWNER/SLUG] [--resume-from-dataset OWNER/SLUG] [--resume] [--strict-identity]
#   --reuse-session NAME  use the existing session NAME (skips `colab new` and the dataset prepare step; the package,
#                         params and results steps still run). Equivalent to --session NAME plus reuse.
#   --persist-dataset S   VM publishes the per-epoch resume state (every >= 8 min, when it changed) as versions of the
#                         private Kaggle dataset S (e.g. ozgung/knee-colab-resume-fold0; "{K}" = fold number).
#   --resume-from-dataset S  prepare downloads the latest version of S into the VM workdir, verifies it loads, and the
#                         trainer continues from that epoch. Implies --persist-dataset S unless that is given too.
#   --resume              continue from a raptor_resume_fold{K}.pt already on the VM (e.g. with --reuse-session).
#   --strict-identity     abort a resume whose saved state has no identity block (old format). A present block (fold, script,
#                         labels, folds, exclusions, config hashes) is always compared and a mismatch always aborts.
#   Persistence: old dataset versions are kept; READY needs status ready + listing == payload_manifest (name, size, time);
#   after training the run script prints FINAL BACKUP VERIFIED epoch N version V (or UNVERIFIED -> rc 1). Status JSON on the VM.
#   Transport loss: if the driver log is silent for 20 min AND `colab sessions` no longer lists the session, the exec is
#   killed and the driver exits 75 so a wrapper can relaunch with --resume-from-dataset.
#   (TRACKA_WATCHDOG_INTERVAL_S=120 and TRACKA_WATCHDOG_STALL_S=1200 override the watchdog timing.)
#
# Env: KAGGLE_JSON (default ~/.kaggle/kaggle.json), PKG_DIR (default outputs/kaggle/trackA-convnext-20261009).
#
# TOKEN HANDLING: this script uploads your Kaggle API token to /content/kaggle.json on the EPHEMERAL Colab VM
# (mode 600; copied to /content/.kaggle/ by the prepare script) so the VM can download the datasets. The token
# is only ever referenced by path; it is never echoed, logged, or passed on a command line. The VM is stopped
# at the end (also on error/interrupt via a trap) unless --keep is given; with --keep the token stays on the
# running VM until you run `colab stop -s NAME`.
#
# Never uses colab repl/console/auth/drivemount (interactive). Does not commit or touch tracked files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="${PKG_DIR:-$ROOT/outputs/kaggle/trackA-convnext-20261009}"
KAGGLE_JSON="${KAGGLE_JSON:-$HOME/.kaggle/kaggle.json}"
PKG_FILES=(tracka_entry.py train_knee_patched.py train_knee.patch train_knee_original.py manifest.json folds_v0.json excluded_ids.json)
REMOTE_PKG=/content/trackA_pkg

SESSION="" FOLD_A="" FOLD_B="" EPOCHS="" TIME_BUDGET="" SMOKE=0 KEEP=0 DRY=0 EXEC_TIMEOUT_H=22
ORIG_ARGS=("$@")
PERSIST_DS="" RESUME_DS="" RESUME_LOCAL=0 STRICT_IDENTITY=0
ARCH="" RES="" KTRAIN="" KEVAL="" BS="" LABELS_FILE="" GRAD_CKPT=0 WIN_CHUNK="" ACCUM="" REUSE=0

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --session) SESSION="${2-}"; shift 2 ;;
    --fold-a) FOLD_A="${2-}"; shift 2 ;;
    --fold-b) FOLD_B="${2-}"; shift 2 ;;
    --epochs) EPOCHS="${2-}"; shift 2 ;;
    --time-budget-h) TIME_BUDGET="${2-}"; shift 2 ;;
    --exec-timeout-h) EXEC_TIMEOUT_H="${2-}"; shift 2 ;;
    --arch) ARCH="${2-}"; shift 2 ;;
    --res) RES="${2-}"; shift 2 ;;
    --k) KTRAIN="${2-}"; shift 2 ;;
    --k-eval) KEVAL="${2-}"; shift 2 ;;
    --bs) BS="${2-}"; shift 2 ;;
    --labels-file) LABELS_FILE="${2-}"; shift 2 ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --grad-ckpt) GRAD_CKPT=1; shift ;;
    --win-chunk) WIN_CHUNK="${2-}"; shift 2 ;;
    --accum) ACCUM="${2-}"; shift 2 ;;
    --reuse-session) SESSION="${2-}"; REUSE=1; shift 2 ;;
    --persist-dataset) PERSIST_DS="${2-}"; shift 2 ;;
    --resume-from-dataset) RESUME_DS="${2-}"; shift 2 ;;
    --resume) RESUME_LOCAL=1; shift ;;
    --strict-identity) STRICT_IDENTITY=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --keep|--no-stop) KEEP=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[[ "$SESSION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--session NAME required (letters, digits, - _)" >&2; exit 2; }
[[ "$FOLD_A" =~ ^[0-4]$ ]] || { echo "--fold-a must be 0..4" >&2; exit 2; }
[[ -z "$FOLD_B" || "$FOLD_B" =~ ^[0-4]$ ]] || { echo "--fold-b must be 0..4 or empty" >&2; exit 2; }
[ "$FOLD_B" != "$FOLD_A" ] || { echo "--fold-a and --fold-b must differ" >&2; exit 2; }
[[ -z "$EPOCHS" || "$EPOCHS" =~ ^[0-9]+$ ]] || { echo "--epochs must be an integer" >&2; exit 2; }
[[ -z "$TIME_BUDGET" || "$TIME_BUDGET" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "--time-budget-h must be a number" >&2; exit 2; }
[[ -z "$ARCH" || "$ARCH" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "--arch must be a timm model name" >&2; exit 2; }
[[ -z "$LABELS_FILE" || "$LABELS_FILE" =~ ^[A-Za-z0-9_.-]+\.parquet$ ]] || { echo "--labels-file must be a bare *.parquet file name" >&2; exit 2; }
for kv in "res:$RES" "k:$KTRAIN" "k-eval:$KEVAL" "bs:$BS" "win-chunk:$WIN_CHUNK" "accum:$ACCUM"; do
  [[ -z "${kv#*:}" || "${kv#*:}" =~ ^[0-9]+$ ]] || { echo "--${kv%%:*} must be a non-negative integer" >&2; exit 2; }
done
for kv in "persist-dataset:$PERSIST_DS" "resume-from-dataset:$RESUME_DS"; do
  [[ -z "${kv#*:}" || "${kv#*:}" =~ ^[a-z0-9_-]+/[a-z0-9{}-]{3,50}$ ]] || { echo "--${kv%%:*} must look like owner/lowercase-slug" >&2; exit 2; }
done
[ -n "$PERSIST_DS" ] || PERSIST_DS="$RESUME_DS"
[[ "$EXEC_TIMEOUT_H" =~ ^[0-9]+$ ]] || { echo "--exec-timeout-h must be an integer" >&2; exit 2; }

OUT="$ROOT/outputs/colab/trackA-$SESSION"
PREPARE_PY="$ROOT/scripts/colab_tracka_prepare.py"
RUN_PY="$ROOT/scripts/colab_tracka_run.py"
PERSIST_PY="$ROOT/scripts/colab_tracka_persist.py"
PARAMS_LOCAL="$(mktemp -t tracka_params.XXXXXX 2>/dev/null || echo /tmp/tracka_params.json)"
[ "$DRY" = 1 ] && PARAMS_LOCAL="/tmp/tracka_params.XXXXXX"

# Params JSON for the run script (exec cannot pass env). Only validated values reach it.
PARAMS_JSON="{\"FOLD_A\": \"$FOLD_A\", \"FOLD_B\": \"$FOLD_B\", \"SMOKE\": \"$SMOKE\""
[ -n "$EPOCHS" ] && PARAMS_JSON+=", \"EPOCHS\": \"$EPOCHS\""
[ -n "$TIME_BUDGET" ] && PARAMS_JSON+=", \"TIME_BUDGET_H\": \"$TIME_BUDGET\""
[ -n "$ARCH" ] && PARAMS_JSON+=", \"ARCH\": \"$ARCH\""
[ -n "$RES" ] && PARAMS_JSON+=", \"RES\": \"$RES\""
[ -n "$KTRAIN" ] && PARAMS_JSON+=", \"K\": \"$KTRAIN\""
[ -n "$KEVAL" ] && PARAMS_JSON+=", \"K_EVAL\": \"$KEVAL\""
[ -n "$BS" ] && PARAMS_JSON+=", \"BS\": \"$BS\""
[ -n "$LABELS_FILE" ] && PARAMS_JSON+=", \"LABELS_FILE\": \"$LABELS_FILE\""
[ "$GRAD_CKPT" = 1 ] && PARAMS_JSON+=", \"GRAD_CKPT\": \"1\""
[ -n "$WIN_CHUNK" ] && PARAMS_JSON+=", \"WIN_CHUNK\": \"$WIN_CHUNK\""
[ -n "$ACCUM" ] && PARAMS_JSON+=", \"ACCUM\": \"$ACCUM\""
[ -n "$PERSIST_DS" ] && PARAMS_JSON+=", \"RESUME_DATASET\": \"$PERSIST_DS\""
[ "${SKIP_EXPORT:-0}" = 1 ] && PARAMS_JSON+=", \"SKIP_EXPORT\": \"1\""
[ -n "$RESUME_DS" ] && PARAMS_JSON+=", \"RESUME_FROM_DATASET\": \"$RESUME_DS\", \"RESUME\": \"1\""
[ "$RESUME_LOCAL" = 1 ] && [ -z "$RESUME_DS" ] && PARAMS_JSON+=", \"RESUME\": \"1\""
[ "$STRICT_IDENTITY" = 1 ] && PARAMS_JSON+=", \"STRICT_IDENTITY\": \"1\""
PARAMS_JSON+="}"

# Packs the result files into one tar on the VM (colab download moves one file at a time).
read -r -d '' PACK_PY <<'PYEOF' || true
import glob, tarfile, os
pats = ["oof_fold*.csv", "run_fold*.json", "log_fold*.txt", "log_export_fold*.txt", "train_meta_fold*.json",
        "raptor_ft_fold*_swa.pt", "raptor_ft_fold*.pt", "raptor_ft_fold*.json", "persist_fold*.log"]
files = sorted({f for p in pats for f in glob.glob("/content/trackA/" + p)} | set(glob.glob("/content/persist_status_fold*.json")))
with tarfile.open("/content/trackA_results.tar", "w") as t:
    for f in files:
        t.add(f, arcname=os.path.basename(f))
print("packed", len(files), "files:", [os.path.basename(f) for f in files])
PYEOF

now() { date +%s; }
STEP_NAME="" STEP_T0=0
step_begin() { STEP_NAME="$1"; STEP_T0=$(now); echo; echo "=== [$(date +%H:%M:%S)] $1"; }
step_end() { local d=$(( $(now) - STEP_T0 )); printf '=== done: %s (elapsed %dm%02ds)\n' "$STEP_NAME" $((d/60)) $((d%60)); }

# run CMD...  : prints the command in dry-run mode, otherwise executes it.
run() {
  if [ "$DRY" = 1 ]; then printf 'DRY-RUN: '; printf '%q ' "$@"; echo; return 0; fi
  "$@"
}
# run_logged LOG CMD... : like run, but tees combined output into LOG.
run_logged() {
  local log="$1"; shift
  if [ "$DRY" = 1 ]; then printf 'DRY-RUN: '; printf '%q ' "$@"; printf '2>&1 | tee %q\n' "$log"; return 0; fi
  "$@" 2>&1 | tee "$log"
}
# run_py_stdin CODE CMD... : pipe python source to a colab exec.
run_py_stdin() {
  local code="$1"; shift
  if [ "$DRY" = 1 ]; then
    printf 'DRY-RUN: '; printf '%q ' "$@"; echo '<<< (inline python:'
    printf '%s\n' "$code" | sed 's/^/            /'; echo '            )'; return 0
  fi
  printf '%s\n' "$code" | "$@"
}

# session_listed: success only if `colab sessions` itself succeeds AND lists the session.
session_listed() { local o; o=$(colab sessions 2>/dev/null) || return 1; printf '%s\n' "$o" | grep -qF -- "$SESSION"; }
# session_gone: success only if `colab sessions` succeeds and does NOT list the session (an erroring CLI is "unknown").
session_gone() { local o; o=$(colab sessions 2>&1) || return 1; ! printf '%s\n' "$o" | grep -qF -- "$SESSION"; }

# print_last_ready WHY: tell the operator which epoch the resume dataset is known to hold READY, from the local copy of
# persist_status_fold*.json if one was downloaded, else from the READY lines/status heartbeats in the driver log.
print_last_ready() {
  local f found=0
  echo "LAST READY RESUME EPOCH ($1):"
  for f in "$OUT"/persist_status_fold*.json; do
    [ -f "$f" ] || continue
    found=1
    python3 - "$f" <<'PYEOF' || echo "  $f: unreadable"
import json, sys
st = json.load(open(sys.argv[1]))
print(f"  {sys.argv[1].rsplit('/', 1)[-1]}: VERIFIED epoch {st.get('verified_epoch')} (version {st.get('verified_version')}), "
      f"accepted epoch {st.get('accepted_epoch')} (version {st.get('accepted_version')}), "
      f"verification_attempts {st.get('verification_attempts')}, last_error {st.get('last_error')!r}, at {st.get('timestamp')}")
PYEOF
  done
  if [ "$found" = 0 ]; then
    echo "  no local persist_status_fold*.json; from the driver log ($RUN_LOG):"
    { grep -E 'PUBLISHED .* READY|PUBLISH-UNVERIFIED|FINAL BACKUP|persist status fold' "$RUN_LOG" 2>/dev/null | tail -n 4 | cut -c1-260; } || true
    grep -qE 'PUBLISHED .* READY|persist status fold' "$RUN_LOG" 2>/dev/null || echo "  none found: no version was confirmed READY in this log"
  fi
}

CREATED=0
cleanup() {
  local rc=$?
  trap - EXIT
  [ -n "${PARAMS_LOCAL:-}" ] && [ "$DRY" != 1 ] && rm -f "$PARAMS_LOCAL"
  if [ "$CREATED" = 1 ] && [ "$DRY" = 1 ]; then
    [ "$KEEP" = 1 ] && echo "DRY-RUN: (--keep) no colab stop" || echo "DRY-RUN: colab stop -s $SESSION   # via EXIT trap, also on error"
  elif [ "$CREATED" = 1 ]; then
    if [ "$KEEP" = 1 ]; then
      if session_listed; then
        echo "NOTE: --keep given; 'colab sessions' still lists '$SESSION', which holds your Kaggle token at /content/kaggle.json." >&2
        echo "      Release it with: colab stop -s $SESSION" >&2
      else
        echo "NOTE: --keep given, but 'colab sessions' does not confirm '$SESSION' is running (gone or unreachable); nothing to release." >&2
      fi
    else
      echo "=== stopping session $SESSION" >&2
      colab stop -s "$SESSION" || echo "WARNING: colab stop failed; run 'colab stop -s $SESSION' / 'colab sessions' manually" >&2
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Track A Colab run: reuse=$REUSE session=$SESSION fold_a=$FOLD_A fold_b='${FOLD_B}' epochs='${EPOCHS:-default}' smoke=$SMOKE keep=$KEEP dry_run=$DRY"
echo "package=$PKG_DIR  results -> $OUT"
# Ownership: scripts/colab_tracka_folds.sh writes this lock (pid, sessions, start time) while it owns the fold loop.
LOCK_FILE="${TRACKA_LOCK_FILE:-$ROOT/outputs/colab/.tracka_folds.lock}"
if [ -f "$LOCK_FILE" ]; then
  LOCK_PID=$(sed -n 's/^pid=//p' "$LOCK_FILE" | head -n 1)
  if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then LOCK_STATE=live; else LOCK_STATE=STALE; fi
  echo "lock owner ($LOCK_STATE): $(tr '\n' ' ' < "$LOCK_FILE") [$LOCK_FILE]"
else
  echo "lock owner: none (driver run by hand; no fold loop holds $LOCK_FILE)"
fi
if [ "$DRY" != 1 ]; then
  command -v colab >/dev/null || { echo "colab CLI not found" >&2; exit 2; }
  [ -f "$KAGGLE_JSON" ] || { echo "Kaggle token not found at $KAGGLE_JSON (set KAGGLE_JSON)" >&2; exit 2; }
  for f in "${PKG_FILES[@]}"; do [ -f "$PKG_DIR/$f" ] || { echo "missing package file $PKG_DIR/$f" >&2; exit 2; }; done
  for f in "$PREPARE_PY" "$RUN_PY" "$PERSIST_PY"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
  mkdir -p "$OUT"
else
  [ -f "$KAGGLE_JSON" ] || echo "(dry-run note: $KAGGLE_JSON does not exist here; a real run would abort)"
fi
T_ALL=$(now)

# (a) allocate
if [ "$REUSE" = 1 ]; then
  step_begin "a. reuse existing session $SESSION (no colab new)"
  if [ "$DRY" != 1 ] && ! colab sessions 2>/dev/null | grep -qF -- "$SESSION"; then
    echo "ERROR: session '$SESSION' not found in 'colab sessions'" >&2; exit 1
  fi
  [ "$DRY" = 1 ] && echo "DRY-RUN: colab sessions | grep $SESSION   # must exist"
  CREATED=1   # the EXIT trap still stops it unless --keep/--no-stop
else
  step_begin "a. allocate A100 session"
  if ! run colab new -s "$SESSION" --gpu A100; then
    echo "ERROR: 'colab new' failed. A 400 response means this account has no A100 entitlement/quota (try --gpu T4 manually, or check Colab compute units)." >&2
    exit 1
  fi
  CREATED=1
fi
step_end

# (b) upload package + token + params
step_begin "b. upload package, params and Kaggle token (token is never printed)"
run_py_stdin "import os; os.makedirs('$REMOTE_PKG', exist_ok=True)" colab exec -s "$SESSION" --timeout 120
for f in "${PKG_FILES[@]}"; do run colab upload -s "$SESSION" "$PKG_DIR/$f" "$REMOTE_PKG/$f"; done
run colab upload -s "$SESSION" "$PERSIST_PY" /content/colab_tracka_persist.py
if [ "$DRY" = 1 ]; then
  echo "DRY-RUN: write $PARAMS_JSON to a local tmpfile"
else
  printf '%s\n' "$PARAMS_JSON" > "$PARAMS_LOCAL"
fi
run colab upload -s "$SESSION" "$PARAMS_LOCAL" /content/tracka_params.json
if [ "$REUSE" != 1 ]; then
  run colab upload -s "$SESSION" "$KAGGLE_JSON" /content/kaggle.json
  run_py_stdin "import os; os.chmod('/content/kaggle.json', 0o600)" colab exec -s "$SESSION" --timeout 120
fi
step_end

# (c) VM-side preparation (downloads ~22 GB of datasets)
CAFF=(); command -v caffeinate >/dev/null && CAFF=(caffeinate -i)
if [ "$REUSE" = 1 ] && [ -z "$RESUME_DS" ]; then
  step_begin "c. prepare VM: skipped (--reuse-session; corpus already downloaded)"
else
  step_begin "c. prepare VM (kaggle downloads, hash checks${RESUME_DS:+, restore resume state from $RESUME_DS})"
  if [ "$DRY" != 1 ]; then : > "$OUT/driver_prepare.log"; fi
  run_logged "$OUT/driver_prepare.log" ${CAFF[@]+"${CAFF[@]}"} colab exec -s "$SESSION" -f "$PREPARE_PY" --timeout 10800
  if [ "$DRY" != 1 ] && ! grep -q '^TRACKA_PREPARE_OK' "$OUT/driver_prepare.log"; then
    echo "ERROR: prepare did not report TRACKA_PREPARE_OK (see $OUT/driver_prepare.log)" >&2; exit 1
  fi
fi
step_end

# (d) training (+ export). A failure here still falls through to the download step to rescue partial results.
step_begin "d. train + export (exec timeout ${EXEC_TIMEOUT_H} h)"
RUN_RC=0
RUN_LOG="$OUT/driver_run_fold${FOLD_A}${FOLD_B:+_$FOLD_B}.log"
WD_INTERVAL="${TRACKA_WATCHDOG_INTERVAL_S:-120}" WD_STALL="${TRACKA_WATCHDOG_STALL_S:-1200}"
LOSS_MARK="$OUT/.transport_loss.$$"
EXEC_PID="" WD_PID=""
# Watchdog (background): if the driver log has not grown for WD_STALL s AND `colab sessions` no longer lists the
# session, kill the exec and leave a marker; the driver then exits 75 (TRANSPORT LOSS).
watchdog() {
  local last_size=-1 last_change size last_line
  last_change=$(now)
  while sleep "$WD_INTERVAL"; do
    size=$(wc -c < "$RUN_LOG" 2>/dev/null | tr -d ' ')
    if [ "$size" != "$last_size" ]; then last_size="$size"; last_change=$(now); continue; fi
    if [ $(( $(now) - last_change )) -ge "$WD_STALL" ] && session_gone; then
      last_line=$(grep -v '^[[:space:]]*$' "$RUN_LOG" | tail -n 1 | cut -c1-200) || true
      echo "TRANSPORT LOSS: session gone at $(date +%H:%M:%S); last log line: $last_line" | tee "$LOSS_MARK"
      pkill -TERM -P "$EXEC_PID" 2>/dev/null || true; kill -TERM "$EXEC_PID" 2>/dev/null || true
      sleep 5
      pkill -KILL -P "$EXEC_PID" 2>/dev/null || true; kill -KILL "$EXEC_PID" 2>/dev/null || true
      return 0
    fi
  done
}
if [ "$DRY" = 1 ]; then
  run_logged "$RUN_LOG" ${CAFF[@]+"${CAFF[@]}"} colab exec -s "$SESSION" -f "$RUN_PY" --timeout $((EXEC_TIMEOUT_H * 3600))
  echo "DRY-RUN: watchdog every ${WD_INTERVAL}s: log silent >= ${WD_STALL}s AND session missing from 'colab sessions' => kill exec, exit 75"
else
  : > "$RUN_LOG"; rm -f "$LOSS_MARK"
  ${CAFF[@]+"${CAFF[@]}"} colab exec -s "$SESSION" -f "$RUN_PY" --timeout $((EXEC_TIMEOUT_H * 3600)) > >(tee "$RUN_LOG") 2>&1 &
  EXEC_PID=$!
  watchdog &
  WD_PID=$!
  wait "$EXEC_PID" || RUN_RC=$?
  kill "$WD_PID" 2>/dev/null || true; wait "$WD_PID" 2>/dev/null || true
  sleep 1   # let tee flush the log
  if [ -f "$LOSS_MARK" ]; then
    RUN_RC=75
  elif [ "$RUN_RC" != 0 ] && ! grep -q '^TRACKA_RUN_OK' "$RUN_LOG" && session_gone; then
    echo "TRANSPORT LOSS: session gone at $(date +%H:%M:%S) (exec exited rc=$RUN_RC); last log line: $(grep -v '^[[:space:]]*$' "$RUN_LOG" | tail -n 1 | cut -c1-200)"
    RUN_RC=75
  fi
  rm -f "$LOSS_MARK"
fi
if [ "$DRY" != 1 ] && [ "$RUN_RC" = 0 ] && ! grep -q '^TRACKA_RUN_OK' "$RUN_LOG"; then
  echo "ERROR: run script did not report TRACKA_RUN_OK (see $RUN_LOG)" >&2; RUN_RC=1
fi
step_end

# (e) results (attempted even if (d) failed)
step_begin "e. download results to $OUT"
if [ "$RUN_RC" = 75 ]; then
  echo "skipped: the session is gone (transport loss); nothing to download. The VM persister's last READY version is the recovery point."
  [ "$DRY" = 1 ] || print_last_ready "transport loss, rc 75"
elif run_py_stdin "$PACK_PY" colab exec -s "$SESSION" --timeout 900 \
   && run colab download -s "$SESSION" /content/trackA_results.tar "$OUT/trackA_results.tar"; then
  if [ "$DRY" = 1 ]; then echo "DRY-RUN: tar -xf $OUT/trackA_results.tar -C $OUT && rm $OUT/trackA_results.tar"
  else
    tar -xf "$OUT/trackA_results.tar" -C "$OUT" && rm -f "$OUT/trackA_results.tar"; ls -la "$OUT"
    ls "$OUT"/persist_status_fold*.json >/dev/null 2>&1 && print_last_ready "persist status downloaded with the results"
  fi
else
  echo "WARNING: result download failed; session left to the trap (use --keep and retry by hand if needed)" >&2
  if session_gone; then
    echo "session '$SESSION' not found at download time (treated as lost)" >&2
    print_last_ready "session not found at download time"
    # Training finished and the run script already confirmed the final backup (TRACKA_RUN_OK follows FINAL BACKUP VERIFIED),
    # so this is classified transport loss: the loop reads the dataset's verified epoch instead of treating it as a failure.
    if [ "$RUN_RC" = 0 ]; then
      echo "TRANSPORT LOSS: session lost at download time after a verified final backup; classified rc 75" >&2
      RUN_RC=75
    fi
  fi
  [ "$RUN_RC" = 0 ] && RUN_RC=1
fi
step_end

# (f) stop is done by the EXIT trap (also on error), unless --keep.
D=$(( $(now) - T_ALL )); printf '\nTotal elapsed %dh%02dm%02ds; run status rc=%d\n' $((D/3600)) $((D%3600/60)) $((D%60)) "$RUN_RC"
if [ "$RUN_RC" = 75 ]; then
  echo "TRANSPORT LOSS: relaunch on a NEW session with the same arguments plus --resume-from-dataset ${PERSIST_DS:-OWNER/SLUG}"
  RELAUNCH=(); skip=0
  for arg in "${ORIG_ARGS[@]}"; do
    if [ "$skip" = 1 ]; then skip=0; continue; fi
    case "$arg" in --session|--reuse-session|--resume-from-dataset) skip=1 ;; --resume) ;; *) RELAUNCH+=("$arg") ;; esac
  done
  printf '  scripts/colab_tracka.sh '; printf '%q ' ${RELAUNCH[@]+"${RELAUNCH[@]}"}
  echo "--session NEW_NAME --resume-from-dataset ${PERSIST_DS:-OWNER/SLUG}"
fi
exit "$RUN_RC"
