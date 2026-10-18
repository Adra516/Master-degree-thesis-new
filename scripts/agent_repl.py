#!/usr/bin/env python3
"""agent_repl — terminal client for vlm_agent_node's run_task service.

Each line typed in is one task, sent to /{ns}/agent/run_task and run to
completion by the VLM tool-use loop. Tasks share ONE ongoing
conversation inside vlm_agent_node (never reset), so a later task can refer
back to what an earlier one found — this script is just a thin front end for
that service call, not a conversation of its own.

Usage (vlm_agent_node already running, e.g. via locobot.launch):
    rosrun locobot_prog agent_repl.py
    > take the bottle, has it got a marker on it?
    ...
    > what did you find?
Ctrl-D or 'exit' to quit.
"""
import json
import sys

import rospy

from locobot_prog.srv import RunTask


def main():
    rospy.init_node("agent_repl", anonymous=True, disable_signals=True)
    ns = rospy.get_param("~robot_name", "locobot")
    service = f"/{ns}/agent/run_task"
    print(f"Waiting for {service} ...")
    rospy.wait_for_service(service)
    run_task = rospy.ServiceProxy(service, RunTask)
    print("Ready. Type a task (Ctrl-D or 'exit' to quit).")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        try:
            r = run_task(task=line)
        except rospy.ServiceException as e:
            print(f"[error] {e}")
            continue
        print(r.answer)
        try:
            plan = json.loads(r.plan_json)
            print(f"[plan] {json.dumps(plan, indent=2)}")
        except (ValueError, TypeError):
            pass


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        sys.exit(0)
