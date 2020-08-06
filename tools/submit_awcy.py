#!/usr/bin/env python3

from __future__ import print_function

import argparse
import os
import subprocess
import sys
from datetime import datetime

import requests


# our timestamping function, accurate to milliseconds
# (remove [:-3] to display microseconds)
def GetTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


if "check_output" not in dir(subprocess):  # duck punch it in!

    def f(*popenargs, **kwargs):
        if "stdout" in kwargs:
            raise ValueError(
                "stdout argument not allowed, it will be overridden."
            )
        process = subprocess.Popen(
            stdout=subprocess.PIPE, *popenargs, **kwargs
        )
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)
        return output

    subprocess.check_output = f

key = None
try:
    with open("secret_key", "r") as keyfile:
        key = keyfile.read().strip()
except:
    print(GetTime(), "Could not open your secret_key file!")
    print(
        GetTime(),
        "Please create a file called secret_key in the current directory",
    )
    print(GetTime(), "containing your AreWeCompressedYet key.")
    sys.exit(1)

parser = argparse.ArgumentParser(
    description="Submit test to arewecompressedyet.com"
)
parser.add_argument("-branch", default=None)
parser.add_argument("-commit", default=None)
parser.add_argument("-prefix", default=None)
parser.add_argument("-master", action="store_true", default=False)
parser.add_argument("-set", default="objective-1-fast")
parser.add_argument("-nick", default=None)
parser.add_argument("-qualities", default=None, nargs="+", type=int)
parser.add_argument("-extra_options", default="")
args = parser.parse_args()

if args.branch is None:
    try:
        args.branch = (
            subprocess.check_output(
                "git symbolic-ref -q --short HEAD", shell=True
            )
            .strip()
            .decode("utf-8")
        )
    except:
        args.branch = None

if args.prefix is None:
    args.prefix = args.branch

if args.commit is None:
    commit = (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .strip()
        .decode("utf-8")
    )
else:
    commit = args.commit
short_arg = "git rev-parse --short " + commit
short = subprocess.check_output(short_arg, shell=True).strip().decode("utf-8")

date = GetTime().strip()
date_short = (
    date.split()[0] + "_" + date.split()[1].split(".")[0].replace(":", "")
)
prefix = args.prefix
is_master = args.master
run_id = prefix + "-" + date_short + "-" + short

params = {
    "run_id": run_id,
    "commit": commit,
    "master": is_master,
    "key": key,
    "task": args.set,
    "codec": "rav1e",
    "nick": args.nick,
    "extra_options": args.extra_options,
}

if args.qualities is not None:
    params["qualities"] = " ".join(map(str, args.qualities))

print(GetTime(), "Creating run " + run_id)
r = requests.post("https://beta.arewecompressedyet.com/submit/job", params)
print(GetTime(), r)
