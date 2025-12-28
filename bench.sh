#!/bin/bash

# Benchmark script - runs each browser 3 times and reports results

RUNS=3
WAIT_TIME=15

# Kill any existing processes
cleanup() {
  pkill -f "Safari Technology Preview" 2>/dev/null
  pkill -f "Chrome Canary" 2>/dev/null
  pkill -f "Firefox Nightly" 2>/dev/null
  pkill -f "websocat" 2>/dev/null
}

cleanup
sleep 1

# Start benchmark server in background, capture output
RESULTS=$(mktemp)
pnpm bench:server > "$RESULTS" 2>&1 &
SERVER_PID=$!
sleep 2

echo "Running benchmarks ($RUNS runs per browser, ${WAIT_TIME}s each)..."
echo

# Chrome
echo "=== Chrome ==="
for i in $(seq 1 $RUNS); do
  echo -n "  Run $i... "
  /Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary \
    --user-data-dir=/tmp/chrome-bench --no-first-run \
    'http://localhost:5173/?benchmark' 2>/dev/null &
  PID=$!
  sleep $WAIT_TIME
  kill $PID 2>/dev/null
  wait $PID 2>/dev/null
  pkill -f "Chrome Canary" 2>/dev/null
  echo "done"
  sleep 1
done

# Safari
echo "=== Safari ==="
for i in $(seq 1 $RUNS); do
  echo -n "  Run $i... "
  open -n -a 'Safari Technology Preview' 'http://localhost:5173/?benchmark'
  sleep $WAIT_TIME
  pkill -f "Safari Technology Preview" 2>/dev/null
  echo "done"
  sleep 1
done

# Firefox - create profile with prefs to suppress dialogs
FF_PROFILE=/tmp/firefox-bench
rm -rf "$FF_PROFILE" && mkdir -p "$FF_PROFILE"
cat > "$FF_PROFILE/user.js" << 'EOF'
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.shell.didSkipDefaultBrowserCheckOnFirstRun", true);
user_pref("browser.shell.skipDefaultBrowserCheckOnFirstRun", true);
user_pref("toolkit.startup.max_resumed_crashes", -1);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("browser.rights.3.shown", true);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);
EOF

echo "=== Firefox ==="
for i in $(seq 1 $RUNS); do
  echo -n "  Run $i... "
  /Applications/Firefox\ Nightly.app/Contents/MacOS/firefox \
    -no-remote --new-instance -profile "$FF_PROFILE" \
    'http://localhost:5173/?benchmark' 2>/dev/null &
  PID=$!
  sleep $WAIT_TIME
  kill $PID 2>/dev/null
  wait $PID 2>/dev/null
  pkill -f "Firefox Nightly" 2>/dev/null
  echo "done"
  sleep 1
done

# Kill server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo
echo "=========================================="
echo "RESULTS"
echo "=========================================="
echo

# Parse results using grep/sed (BSD compatible)
parse_browser() {
  local browser=$1
  local data=$(grep -A15 "^\[$browser\] Warmup:" "$RESULTS")

  # Extract all speedups for this browser
  local per_speeds=$(echo "$data" | grep -A3 "Per-Iteration Timing" | grep "Speedup:" | sed 's/.*Speedup: \([0-9.]*\)x.*/\1/')
  local wall_speeds=$(echo "$data" | grep -A3 "Wall-Clock Timing" | grep "Speedup:" | sed 's/.*Speedup: \([0-9.]*\)x.*/\1/')

  # Extract median times
  local frag_per=$(echo "$data" | grep -A1 "Per-Iteration Timing" | grep "Fragment:" | sed 's/.*median=\([0-9.]*\).*/\1/')
  local comp_per=$(echo "$data" | grep -A2 "Per-Iteration Timing" | grep "Compute:" | sed 's/.*median=\([0-9.]*\).*/\1/')
  local frag_wall=$(echo "$data" | grep -A1 "Wall-Clock Timing" | grep "Fragment:" | sed 's/.*median=\([0-9.]*\).*/\1/')
  local comp_wall=$(echo "$data" | grep -A2 "Wall-Clock Timing" | grep "Compute:" | sed 's/.*median=\([0-9.]*\).*/\1/')

  # Get median (middle of 3 sorted values)
  local per_median=$(echo "$per_speeds" | sort -n | sed -n '2p')
  local wall_median=$(echo "$wall_speeds" | sort -n | sed -n '2p')
  local fp_median=$(echo "$frag_per" | sort -n | sed -n '2p')
  local cp_median=$(echo "$comp_per" | sort -n | sed -n '2p')
  local fw_median=$(echo "$frag_wall" | sort -n | sed -n '2p')
  local cw_median=$(echo "$comp_wall" | sort -n | sed -n '2p')

  echo "$browser:"
  echo "  Per-Iteration:  Fragment=${fp_median} us  Compute=${cp_median} us  Speedup=${per_median}x"
  echo "  Wall-Clock:     Fragment=${fw_median} us  Compute=${cw_median} us  Speedup=${wall_median}x"
  echo
}

parse_browser "Chrome"
parse_browser "Safari"
parse_browser "Firefox"

rm "$RESULTS"
cleanup
