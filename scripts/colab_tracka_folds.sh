#!/usr/bin/env bash
# Sequential fold loop around scripts/colab_tracka.sh (the maintained version of the scratch colab_raw_folds.sh).
#
#   scripts/colab_tracka_folds.sh [--continue] FOLD [FOLD ...]
#
# Per fold, up to FOLDS_MAX_ATTEMPTS (3) sessions. After the driver returns:
#   rc 0   -> FOLD DONE.
#   rc 75  -> classified transport loss: download the dataset's payload_manifest.json and resume file (kaggle datasets download -f) and
#             relaunch with --resume-from-dataset --strict-identity only if the verified payload is not complete; if
#             scripts/colab_tracka_verify_payload.py says TRAINED (resume sha256 == manifest, identity == current package, epoch ==
#             EPOCHS-1, history EPOCHS, top-k) the final state is in the dataset; an unverifiable payload STOPs (rc 75): print "FOLD K TRAINED (final state verified in dataset); export pending" and go on.
#   other  -> stop the whole loop (exit with that rc); there is NO generic retry.
# The first attempt of a fold trains FRESH unless --continue is given (then it resumes from the dataset, strict identity).
# Exit status: 0 only if every fold was DONE or TRAINED; otherwise the failing rc (1 when attempts were exhausted).
# Ownership: writes outputs/colab/.tracka_folds.lock (pid, sessions, start time); refuses to start if a live pid holds it;
# removes it on exit. Env overrides (tests): FOLDS_DRIVER, FOLDS_KAGGLE, FOLDS_LOCK_FILE, FOLDS_SLEEP_S, FOLDS_EPOCHS,
# FOLDS_MAX_ATTEMPTS, FOLDS_DS_TEMPLATE, FOLDS_VERIFY, FOLDS_PYTHON, FOLDS_PKG_DIR, FOLDS_LABELS_FILE, FOLDS_DRIVER_ARGS (extra driver args, space separated).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export KAGGLE_JSON="${KAGGLE_JSON:-$HOME/.kaggle/kaggle.json}"
DRIVER="${FOLDS_DRIVER:-$ROOT/scripts/colab_tracka.sh}"
KAGGLE_BIN="${FOLDS_KAGGLE:-$(command -v kaggle || echo /Users/ozgun/miniconda3/bin/kaggle)}"
LOCK_FILE="${FOLDS_LOCK_FILE:-$ROOT/outputs/colab/.tracka_folds.lock}"
SLEEP_S="${FOLDS_SLEEP_S:-60}"
EPOCHS="${FOLDS_EPOCHS:-12}"
MAX_ATTEMPTS="${FOLDS_MAX_ATTEMPTS:-3}"
DS_TEMPLATE="${FOLDS_DS_TEMPLATE:-ozgung/knee-colab-resume-c384r-fold@K@}"
ARCH=coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k RES_PX=384 K_TRAIN=12 K_EVAL=24 BS=8
SESS_PREFIX="${FOLDS_SESSION_PREFIX:-knee-c384r-f}"
CONTINUE_FIRST=0

ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*"; }

if [ "${1-}" = "--continue" ]; then CONTINUE_FIRST=1; shift; fi
[ $# -gt 0 ] || { echo "usage: $0 [--continue] FOLD [FOLD ...]" >&2; exit 2; }
for f in "$@"; do [[ "$f" =~ ^[0-4]$ ]] || { echo "fold must be 0..4, got '$f'" >&2; exit 2; }; done
[[ "$EPOCHS" =~ ^[0-9]+$ ]] && [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || { echo "FOLDS_EPOCHS / FOLDS_MAX_ATTEMPTS must be integers" >&2; exit 2; }

# ---- ownership lock (atomic: write a temp file, hard-link it to the lock name; ln fails if the lock exists)
mkdir -p "$(dirname "$LOCK_FILE")"
SESSIONS=""
for f in "$@"; do for n in $(seq 1 "$MAX_ATTEMPTS"); do SESSIONS+="${SESS_PREFIX}${f}-a${n},"; done; done
write_lock_tmp() {
  local t; t="$(mktemp "$LOCK_FILE.XXXXXX")" || return 1
  { echo "pid=$$"; echo "started=$(ts)"; echo "folds=$*"; echo "sessions=${SESSIONS%,}"; echo "driver=$DRIVER"; } > "$t"
  echo "$t"
}
acquire_lock() {
  local t; t="$(write_lock_tmp "$@")" || return 1
  if ln "$t" "$LOCK_FILE" 2>/dev/null; then rm -f "$t"; return 0; fi
  local owner; owner=$(sed -n 's/^pid=//p' "$LOCK_FILE" 2>/dev/null | head -n 1)
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    rm -f "$t"
    say "REFUSING TO START: lock $LOCK_FILE is held by live pid $owner:"; sed 's/^/    /' "$LOCK_FILE" >&2
    return 1
  fi
  say "removing stale lock (pid ${owner:-unknown} not running)"
  rm -f "$LOCK_FILE"
  if ln "$t" "$LOCK_FILE" 2>/dev/null; then rm -f "$t"; return 0; fi
  rm -f "$t"; say "REFUSING TO START: lost the race for $LOCK_FILE"; return 1
}
acquire_lock "$@" || exit 3
trap 'rm -f "$LOCK_FILE"' EXIT
trap 'say "interrupted"; exit 130' INT TERM
say "lock acquired: $LOCK_FILE (pid $$, folds $*, sessions ${SESSIONS%,})"

# ---- verdict on the newest payload stored in the dataset: scripts/colab_tracka_verify_payload.py downloads payload_manifest.json and
# the resume file, checks sha256 against the manifest and the identity block against the CURRENT package, and prints a JSON
# verdict (TRAINED / RESUME / STOP; exit 0 for the first two). A staged status file is never trusted for this decision.
VERIFY="${FOLDS_VERIFY:-$ROOT/scripts/colab_tracka_verify_payload.py}"
PYBIN="${FOLDS_PYTHON:-$( [ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || command -v python3 )}"
PKG_DIR="${FOLDS_PKG_DIR:-$ROOT/outputs/kaggle/trackA-convnext-20261009}"
LABELS_LOCAL="${FOLDS_LABELS_FILE:-$ROOT/data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet}"
payload_verdict() {
  local ds="$1" k="$2"
  "$PYBIN" "$VERIFY" --dataset "$ds" --fold "$k" --epochs "$EPOCHS" --pkg-dir "$PKG_DIR" --labels-file "$LABELS_LOCAL" --kaggle "$KAGGLE_BIN" \
    --expect arch="$ARCH" --expect res="$RES_PX" --expect k="$K_TRAIN" --expect k_eval="$K_EVAL" --expect bs="$BS" --expect grad_ckpt=True 2>/dev/null | tail -n 1
}
verdict_field() { python3 -I -c 'import json,sys
try:
    v=json.loads(sys.argv[1]); x=v.get(sys.argv[2]); print("" if x is None else (" | ".join(x) if isinstance(x,list) else x))
except Exception:
    print("")' "$1" "$2"; }

# TIME_BUDGET_H (env or a --time-budget-h driver arg) never influences the relaunch decision: only the verified payload does.
unset TIME_BUDGET_H
if [[ " ${FOLDS_DRIVER_ARGS:-} " == *" --time-budget-h "* ]]; then
  say "ignoring --time-budget-h in FOLDS_DRIVER_ARGS (relaunch decisions use the verified payload only)"
  FOLDS_DRIVER_ARGS="$(echo " ${FOLDS_DRIVER_ARGS} " | sed -E 's/ --time-budget-h +[^ ]+ / /')"
fi

FAILED=0
DONE_LIST="" TRAINED_LIST=""
for FOLD in "$@"; do
  DS="${DS_TEMPLATE//@K@/$FOLD}"
  fold_state=""
  attempt=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    SESS="${SESS_PREFIX}${FOLD}-a${attempt}"
    if [ "$attempt" -eq 1 ] && [ "$CONTINUE_FIRST" -ne 1 ]; then
      RES=(--persist-dataset "$DS"); MODE="fresh start"
    else
      RES=(--persist-dataset "$DS" --resume-from-dataset "$DS" --strict-identity); MODE="resume from dataset (strict identity)"
    fi
    say "fold $FOLD attempt $attempt/$MAX_ATTEMPTS session $SESS: launching ($MODE)"
    # shellcheck disable=SC2086
    "$DRIVER" --session "$SESS" --fold-a "$FOLD" --fold-b "" --arch "$ARCH" --res "$RES_PX" --k "$K_TRAIN" --k-eval "$K_EVAL" --bs "$BS" \
      --epochs "$EPOCHS" --labels-file labels_v1_blend.parquet --grad-ckpt --skip-export ${FOLDS_DRIVER_ARGS:-} --time-budget-h 0 "${RES[@]}"
    rc=$?
    say "fold $FOLD attempt $attempt driver rc=$rc"
    if [ "$rc" -eq 0 ]; then
      say "FOLD $FOLD DONE"; fold_state=done; break
    elif [ "$rc" -eq 75 ]; then
      vj="$(payload_verdict "$DS" "$FOLD")"
      vd="$(verdict_field "$vj" verdict)"; ep="$(verdict_field "$vj" epoch)"
      if [ "$vd" != "TRAINED" ] && [ "$vd" != "RESUME" ]; then
        say "FOLD $FOLD STOPPED: rc 75 but the payload in $DS is not verified ($(verdict_field "$vj" reasons)); not relaunching blindly"
        FAILED=75; fold_state=failed; break
      fi
      say "fold $FOLD: dataset payload verified ($vd, epoch $ep of $((EPOCHS - 1)))"
      if [ "$vd" = "TRAINED" ]; then
        say "FOLD $FOLD TRAINED (final state verified in dataset); export pending"
        fold_state=trained; break
      fi
      if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        say "FOLD $FOLD EXHAUSTED: $MAX_ATTEMPTS attempts used, verified epoch $ep < $((EPOCHS - 1))"
        FAILED=1; fold_state=failed; break
      fi
      say "fold $FOLD: transport loss with verified epoch $ep < $((EPOCHS - 1)); relaunching with --resume-from-dataset --strict-identity in ${SLEEP_S}s"
      sleep "$SLEEP_S"
      attempt=$((attempt + 1))
    else
      say "FOLD $FOLD FAILED rc=$rc (not classified transport loss; no retry); stopping the loop"
      FAILED=$rc; fold_state=failed; break
    fi
  done
  case "$fold_state" in
    done) DONE_LIST+=" $FOLD" ;;
    trained) TRAINED_LIST+=" $FOLD" ;;
    *) [ "$FAILED" -ne 0 ] || FAILED=1; say "LOOP STOPPED at fold $FOLD (exit $FAILED); done:${DONE_LIST:- none} trained-pending-export:${TRAINED_LIST:- none}"; exit "$FAILED" ;;
  esac
done
say "ALL FOLDS LOOP FINISHED: done:${DONE_LIST:- none} trained-pending-export:${TRAINED_LIST:- none}"
exit 0
