#!/bin/sh
# robojev pod 자기 종료 감시 (Task cloud1) — 실행기가 죽어도 pod가 남지 않게 한다.
#
# pod를 만들 때 컨테이너 시작 명령이 이 스크립트를 띄운다(실행기는 아직 접속하지 않았다 — 그래서 감시는 만들 때 함께 심는다).
# POSIX sh로만 쓴다(dash): 파이썬·venv가 깨져도 돈다. 지우는 길은 넷이고 하나라도 될 때까지 끝없이 다시 한다:
#   1) runpodctl pod delete <id>   2) runpodctl remove pod <id> (옛 문법)   3) REST v2 DELETE   4) GraphQL podTerminate
# 셋 다 RunPod가 pod에 넣어 주는 pod 범위 키(RUNPOD_API_KEY)를 쓴다. 키는 로그에 남기지 않는다(set -x 금지, curl은 -sS).
#
# 지우는 조건 (어느 하나라도):
#   deadline      지금 ≥ ROBOJEV_TERMINATE_AT (절대 시각, epoch 초 — 재시작해도 늘어나지 않는다)
#   never-started 감시가 뜬 뒤 ROBOJEV_START_GRACE 초 안에 run 디렉터리에 started 표지가 생기지 않았다 (실행기가 오다 죽었다)
#   done-unfetched finished 표지가 생긴 뒤 ROBOJEV_DONE_GRACE 초가 지났다 (끝난 run을 아무도 가져가 지우지 않았다)
set -u
POD="${RUNPOD_POD_ID:-}"
AT="${ROBOJEV_TERMINATE_AT:-0}"
RUN_DIR="${ROBOJEV_RUN_DIR:-/root/robojev/run}"
START_GRACE="${ROBOJEV_START_GRACE:-3600}"
DONE_GRACE="${ROBOJEV_DONE_GRACE:-1800}"
POLL="${ROBOJEV_WATCHDOG_POLL:-20}"
API="${ROBOJEV_API_BASE:-https://api.runpod.io}"
BOOT=$(date +%s)

log() { echo "[robojev-watchdog $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

terminate_once() {
  if command -v runpodctl >/dev/null 2>&1; then
    if runpodctl pod delete "$POD" >/dev/null 2>&1; then log "지움 요청 성공: runpodctl pod delete"; return 0; fi
    if runpodctl remove pod "$POD" >/dev/null 2>&1; then log "지움 요청 성공: runpodctl remove pod"; return 0; fi
  fi
  if command -v curl >/dev/null 2>&1 && [ -n "${RUNPOD_API_KEY:-}" ]; then
    if curl -fsS -m 30 -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" "$API/v2/pods/$POD" >/dev/null 2>&1; then
      log "지움 요청 성공: REST v2 DELETE"; return 0
    fi
    if curl -fsS -m 30 -X POST -H "Content-Type: application/json" -H "Authorization: Bearer ${RUNPOD_API_KEY}" "$API/graphql" \
         -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD\\\"}) }\"}" >/dev/null 2>&1; then
      log "지움 요청 성공: GraphQL podTerminate"; return 0
    fi
  fi
  return 1
}

terminate_forever() {
  log "pod $POD를 지운다 — 이유: $1"
  n=0
  while :; do
    n=$((n + 1))
    if terminate_once; then
      log "요청 $n번째에 받아들여졌다 — 컨테이너가 곧 멈춘다 (멈추지 않으면 30 s 뒤 다시)"
    else
      log "요청 $n번째 실패 — 30 s 뒤 다시 (돈이 나가는 중이다)"
      # 지우기가 안 되면 멈추기라도 (GPU 요금이 멈춘다; 디스크는 남는다)
      if command -v runpodctl >/dev/null 2>&1; then runpodctl pod stop "$POD" >/dev/null 2>&1 || runpodctl stop pod "$POD" >/dev/null 2>&1; fi
    fi
    sleep 30
  done
}

if [ -z "$POD" ]; then log "RUNPOD_POD_ID가 없다 — 지울 대상을 모른다 (RunPod 밖의 상자인가)"; fi
case "$AT" in ''|*[!0-9]*) log "ROBOJEV_TERMINATE_AT이 정수가 아니다 ($AT) — 곧바로 지운다"; terminate_forever "bad-deadline";; esac
log "무장: 마감 $AT (지금 $BOOT, 남은 $((AT - BOOT)) s) · 시작 유예 $START_GRACE s · 끝난 뒤 유예 $DONE_GRACE s · run $RUN_DIR"
while :; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$AT" ]; then terminate_forever "deadline"; fi
  if [ ! -e "$RUN_DIR/started" ] && [ $((NOW - BOOT)) -ge "$START_GRACE" ]; then terminate_forever "never-started"; fi
  if [ -e "$RUN_DIR/finished" ]; then
    DONE=$(stat -c %Y "$RUN_DIR/finished" 2>/dev/null || echo "$NOW")
    if [ $((NOW - DONE)) -ge "$DONE_GRACE" ]; then terminate_forever "done-unfetched"; fi
  fi
  sleep "$POLL"
done
