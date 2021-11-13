import os
import sys

USEAGE = "Useage: python change_extensions <filename> <fromext> <toext>"

if __name__ == "__main__":
    # extract args
    command, *arg_strings = sys.argv

    # check num args
    if len(arg_strings) != 3:
        print("ERROR: Incorrect number of arguments -- expected 3, got {}".format(len(arg_strings)))
        print(USEAGE) 
        exit(-1)

    # split args
    argdir, fromext, toext = arg_strings

    # validate arg values
    if fromext[0] != "." or toext[0] != ".":
        print("ERROR: extensions must start with a period")
        exit(-1)
    if len(fromext) < 2 or len(toext) < 2:
        print("ERROR: expect extensions to have length at least 1 character after the period")
        exit(-1)
    if not os.path.isdir(argdir):
        print("ERROR: directory /{} not found in current directory".format(argdir))
        exit(-1)

    # change all extensions which match <fromext> to <toext> inside directory <argdir>
    print("Changing all files in /{} with a {} extension to use a {} extension instead...".format(argdir, fromext, toext))
    for filename in os.listdir(argdir):
        path = os.path.join(argdir, filename)
        pre, ext = os.path.splitext(path)
        if ext == fromext:
            os.rename(path, pre + toext)