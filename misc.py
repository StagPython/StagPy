"""miscellaneous definitions"""


def takefield(idx):
    """returns a function returning a field from
    a StagData object"""

    return lambda stagdata: stagdata.fields[idx]
