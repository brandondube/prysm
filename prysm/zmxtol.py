import re

START = '    Trial 	    Criterion 	      Change'
END = 'Number of traceable Monte Carlo files generated:'
start = re.compile(START)
end = re.compile(END)

def extract_mc_trials(file):
    with open(file, 'r') as fid:
        contents = fid.read()

    offset = start.search(contents).end()
    endpt = end.search(contents, pos=offset).start()

    mc_rows = contents[offset:endpt].splitlines()
    return [float(row.split()[1]) for row in mc_rows]
