builtin_interfaces/Time stamp

string run_id
string suite
string contest
string simulator        # empty when --sim not given

uint32 env_n
bool   headless         # headless flag passed to the run

uint32 steps_total
uint32 steps_done       # status=ok
uint32 steps_partial
uint32 steps_failed
uint32 steps_skipped
uint32 steps_in_flight

string[] active_keys    # currently-running step keys
