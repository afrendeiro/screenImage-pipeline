#!/usr/bin/env python

"""
Image processing pipeline
"""

from argparse import ArgumentParser
import os
import sys
import logging
import pandas as pd
import numpy as np
import time
import random
import re
import string
import textwrap
import shutil
import subprocess
import skimage


# TODO: solve pandas chained assignments
pd.options.mode.chained_assignment = None

__author__ = "Andre Rendeiro"
__copyright__ = "Copyright 2014, Andre Rendeiro"
__credits__ = []
__license__ = "GPL2"
__version__ = "0.1"
__maintainer__ = "Andre Rendeiro"
__email__ = "arendeiro@cemm.oeaw.ac.at"
__status__ = "Development"


def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Image processing pipeline.")

    # Global options
    # positional arguments
    # optional arguments
    parser.add_argument("-r", "--project-root", default="/fhgfs/groups/lab_bock/shared/projects/",
                        dest="project_root", type=str,
                        help="""Directory in which the project will reside.
                        Default=/fhgfs/groups/lab_bock/shared/projects/.""")
    parser.add_argument("--html-root", default="/fhgfs/groups/lab_bock/public_html/arendeiro/",
                        dest="html_root", type=str,
                        help="""public_html directory in which bigwig files for the project will reside.
                        Default=/fhgfs/groups/lab_bock/public_html/.""")
    parser.add_argument("--url-root", default="http://www.biomedical-sequencing.at/bocklab/arendeiro/",
                        dest="url_root", type=str,
                        help="""Url mapping to public_html directory where bigwig files for the project will be accessed.
                        Default=http://www.biomedical-sequencing.at/bocklab.""")
    parser.add_argument("--keep-tmp-files", dest="keep_tmp", action="store_true",
                        help="Keep intermediary files. If not it will only preserve final files. Default=False.")
    parser.add_argument("-c", "--cpus", default=16, dest="cpus",
                        help="Number of CPUs to use. Default=16.", type=int)
    parser.add_argument("-m", "--mem-per-cpu", default=2000, dest="mem",
                        help="Memory per CPU to use. Default=2000.", type=int)
    parser.add_argument("-q", "--queue", default="shortq", dest="queue",
                        choices=["develop", "shortq", "mediumq", "longq"],
                        help="Queue to submit jobs to. Default=shortq", type=str)
    parser.add_argument("-t", "--time", default="10:00:00", dest="time",
                        help="Maximum time for jobs to run. Default=10:00:00", type=str)
    parser.add_argument("--user-mail", default="", dest="user_mail",
                        help="User mail address. Default=<submitting user>.", type=str)
    parser.add_argument("-l", "--log-level", default="INFO", dest="log_level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level. Default=INFO.", type=str)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Dry run. Assemble commands, but do not submit jobs to slurm. Default=False.")

    # Sub commands
    subparser = parser.add_subparsers(title="sub-command", dest="command")
    # preprocess
    preprocess_subparser = subparser.add_parser("preprocess")
    preprocess_subparser.add_argument(dest="project_name", help="Project name.", type=str)
    preprocess_subparser.add_argument(dest="csv", help="CSV file with sample annotation.", type=str)
    preprocess_subparser.add_argument("-s", "--stage", default="all", dest="stage",
                                      choices=["all", "bam2fastq", "fastqc", "trimming", "mapping",
                                               "shiftreads", "markduplicates", "removeduplicates",
                                               "indexbam", "qc", "maketracks"],
                                      help="Run only these stages. Default=all.", type=str)

    # preprocess
    preprocess_subparser = subparser.add_parser("stats")
    preprocess_subparser.add_argument(dest="project_name", help="Project name.", type=str)
    preprocess_subparser.add_argument(dest="csv", help="CSV file with sample annotation.", type=str)

    # Parse
    args = parser.parse_args()

    # Logging
    logger = logging.getLogger(__name__)
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(levels[args.log_level])

    # create a file handler
    # (for now in current working dir, in the end copy log to projectDir)
    handler = logging.FileHandler(os.path.join(os.getcwd(), args.project_name + ".log"))
    handler.setLevel(logging.INFO)
    # format logger
    formatter = logging.Formatter(fmt='%(levelname)s: %(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create a stout handler
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.ERROR)
    formatter = logging.Formatter(fmt='%(levelname)s: %(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    stdout.setFormatter(formatter)
    logger.addHandler(stdout)

    # Start main function
    if args.command == "preprocess":
        logger, fr, to = preprocess(args, logger)

    # Exit
    logger.info("Finished and exiting.")

    # Copy log to projectDir
    shutil.copy2(fr, to)

    sys.exit(1)


def checkProjectDirs(args, logger):
    # Directories and paths
    # check args.project_root exists and user has write access
    args.project_root = os.path.abspath(args.project_root)
    logger.debug("Checking if %s directory exists and is writable." % args.project_root)
    if not os.access(args.project_root, os.W_OK):
        logger.error("%s does not exist, or user has no write access.\n\
Use option '-r' '--project-root' to set a non-default project root path." % args.project_root)
        sys.exit(1)

        # check args.html_root exists and user has write access
    htmlDir = os.path.abspath(args.html_root)
    logger.debug("Checking if %s directory exists and is writable." % args.project_root)
    if not os.access(htmlDir, os.W_OK):
        logger.error("%s does not exist, or user has no write access.\n\
Use option '--html-root' to set a non-default html root path." % htmlDir)
        sys.exit(1)

    # project directories
    projectDir = os.path.join(args.project_root, args.project_name)
    dataDir = os.path.join(projectDir, "data")
    resultsDir = os.path.join(projectDir, "results")

    # make relative project dirs
    dirs = [
        projectDir,
        os.path.join(projectDir, "runs"),
        dataDir,
        os.path.join(dataDir, "fastq"),
        os.path.join(dataDir, "fastqc"),
        os.path.join(dataDir, "raw"),
        os.path.join(dataDir, "mapped"),
        os.path.join(dataDir, "coverage"),
        os.path.join(dataDir, "peaks"),
        os.path.join(dataDir, "motifs"),
        resultsDir,
        os.path.join(resultsDir, "plots"),
        htmlDir,
        os.path.join(args.html_root, args.project_name),
        os.path.join(args.html_root, args.project_name, "bigWig")
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # chmod of paths to public_html folder
    html = [
        htmlDir,
        os.path.join(args.html_root, args.project_name),
        os.path.join(args.html_root, args.project_name, "bigWig")
    ]
    for d in html:
        try:
            os.chmod(d, 0755)
        except OSError:
            logger.error("cannot change folder's mode: %s" % d)
            continue

    htmlDir = os.path.join(args.html_root, args.project_name, "bigWig")
    urlRoot = args.url_root + args.project_name + "/bigWig/"

    return (htmlDir, projectDir, dataDir, resultsDir, urlRoot)


def groupWells(plate):
    """
    """
    # ignore some fields in the annotation sheet
    variables = plate.columns.tolist()
    exclude = ["mol", "SourcePlate", "SourceWell", "SourceConc_mM", "Volume_transferred_nl", "DataPath"]
    [variables.pop(variables.index(exc)) for exc in exclude if exc in variables]

    # get well names
    plate["wellName"] = ["_".join([str(j) for j in plate.ix[i][variables]]) for i in plate.index]

    variables.pop(variables.index("Well"))

    for key, values in plate.groupby(variables).groups.items():
        group = plate.ix[values][variables].reset_index(drop=True).ix[0]
        group["imageFiles"] = [os.path.join(path, i) for path in plate.ix[values]["DataPath"]]
        group["sampleName"] = "_".join([str(i) for i in group[variables]])

        # append biological replicate to sample annotation
        plateMerged = plateMerged.append(rep, ignore_index=True)

    # get merged biological replicates -> merged biological replicates
    variables.pop(variables.index("biologicalReplicate"))

    for key, values in plate.groupby(variables).groups.items():
        rep = plate.ix[values][varsName].reset_index(drop=True).ix[0]
        rep["experimentName"] = np.nan
        rep["technicalReplicate"] = 0
        rep["biologicalReplicate"] = 0
        rep["filePath"] = plate.ix[values]["filePath"].tolist()
        rep["sampleName"] = "_".join([str(i) for i in rep[varsName]])

        # append biological replicate to sample annotation
        samplesMerged = samplesMerged.append(rep, ignore_index=True)

    # add field for manual sample pairing
    samplesMerged["controlSampleName"] = None

    # replace sample name with list of sample names for only one sample (original case)
    for i in range(len(samplesMerged)):
        if type(samplesMerged["filePath"][i]) is not list:
            samplesMerged["filePath"][i] = [samplesMerged["filePath"][i]]

    return samplesMerged.sort(["sampleName"])


def getReplicates(samples):
    """
    Returns new sample annotation sheet with provided samples, plus biological replicates
    (merged technical replicates) and merged biological replicates.
    samples - a pandas.DataFrame with sample info.

    """
    # ignore some fields in the annotation sheet
    variables = samples.columns.tolist()
    exclude = ["sampleNumber", "sampleName", "experimentName", "filePath", "controlSampleName"]
    [variables.pop(variables.index(exc)) for exc in exclude if exc in variables]
    varsName = list(variables)

    # get sample names
    samples["sampleName"] = ["_".join([str(j) for j in samples.ix[i][variables]]) for i in samples.index]

    samplesMerged = samples.copy()

    # get merged technical replicates -> biological replicates
    variables.pop(variables.index("technicalReplicate"))

    for key, values in samples.groupby(variables).groups.items():
        rep = samples.ix[values][varsName].reset_index(drop=True).ix[0]
        rep["experimentName"] = np.nan
        rep["technicalReplicate"] = 0
        rep["filePath"] = samples.ix[values]["filePath"].tolist()
        rep["sampleName"] = "_".join([str(i) for i in rep[varsName]])

        # append biological replicate to sample annotation
        samplesMerged = samplesMerged.append(rep, ignore_index=True)

    # get merged biological replicates -> merged biological replicates
    variables.pop(variables.index("biologicalReplicate"))

    for key, values in samples.groupby(variables).groups.items():
        rep = samples.ix[values][varsName].reset_index(drop=True).ix[0]
        rep["experimentName"] = np.nan
        rep["technicalReplicate"] = 0
        rep["biologicalReplicate"] = 0
        rep["filePath"] = samples.ix[values]["filePath"].tolist()
        rep["sampleName"] = "_".join([str(i) for i in rep[varsName]])

        # append biological replicate to sample annotation
        samplesMerged = samplesMerged.append(rep, ignore_index=True)

    # add field for manual sample pairing
    samplesMerged["controlSampleName"] = None

    # replace sample name with list of sample names for only one sample (original case)
    for i in range(len(samplesMerged)):
        if type(samplesMerged["filePath"][i]) is not list:
            samplesMerged["filePath"][i] = [samplesMerged["filePath"][i]]

    return samplesMerged.sort(["sampleName"])


def preprocess(args, logger):
    """
    This takes unmapped Bam files and makes trimmed, aligned, indexed (and shifted if necessary)
    Bam files along with a UCSC browser track.
    """

    logger.info("Starting sample preprocessing.")

    logger.debug("Checking project directories exist and creating if not.")
    htmlDir, projectDir, dataDir, resultsDir, urlRoot = checkProjectDirs(args, logger)

    # Paths to static files on the cluster
    # Other static info

    # Parse sample information
    args.csv = os.path.abspath(args.csv)

    # check if exists and is a file
    if not os.path.isfile(args.csv):
        logger.error("Sample annotation '%s' does not exist, or user has no read access." % args.csv)
        sys.exit(1)

    # read in
    samples = pd.read_csv(args.csv)

    # TODO:
    # Perform checks on the variables given
    # (e.g. genome in genomeIndexes)

    # start pipeline
    projectName = string.join([args.project_name, time.strftime("%Y%m%d-%H%M%S")], sep="_")

    # Get biological replicates and merged biological replicates
    logger.debug("Checking which samples should be merged.")
    samplesMerged = getReplicates(samples)  # <- this is the new annotation sheet as well

    # Preprocess samples
    for sample in range(len(samplesMerged)):
        # Get sample name
        sampleName = samplesMerged["sampleName"][sample]

        # get jobname
        jobName = projectName + "_" + sampleName

        # get intermediate names for files

        # keep track of temporary files
        tempFiles = list()

        # assemble commands
        # get job header
        jobCode = slurmHeader(
            jobName=jobName,
            output=os.path.join(projectDir, "runs", jobName + ".slurm.log"),
            queue=args.queue,
            time=args.time,
            cpusPerTask=args.cpus,
            memPerCpu=args.mem,
            userMail=args.user_mail
        )
        if args.stage in ["all"]:
            # if more than one technical replicate, merge bams
            if len(samplesMerged["filePath"][sample]) > 1:
                jobCode += performWork(
                    input=samplesMerged["filePath"][sample],  # this is a list of sample paths
                    output=sampleName + "_out.png"
                )

        # Remove intermediary files
        if args.stage == "all" and not args.keep_tmp:
            logger.debug("Removing intermediary files")
            for fileName in tempFiles:
                jobCode += removeFile(fileName)

        # Submit job to slurm
        # Get concatenated string with code from all modules
        jobCode += slurmFooter()

        # Output file name
        jobFile = os.path.join(projectDir, "runs", jobName + ".sh")

        with open(jobFile, 'w') as handle:
            handle.write(textwrap.dedent(jobCode))

        # Submit to slurm
        if not args.dry_run:
            logger.info("Submitting jobs to slurm")
            status = slurmSubmitJob(jobFile)

            if status != 0:
                logger.error("Slurm job '%s' not successfull" % jobFile)
                sys.exit(1)
            logger.debug("Project '%s'submission finished successfully." % args.project_name)

    # write original annotation sheet to project folder
    samples.to_csv(os.path.join(projectDir, args.project_name + ".annotation_sheet.csv"), index=False)
    # write annotation sheet with biological replicates to project folder
    samplesMerged.to_csv(os.path.join(projectDir, args.project_name + ".replicates.annotation_sheet.csv"), index=False)

    logger.debug("Finished preprocessing")

    return (logger,
            os.path.join(os.getcwd(), args.project_name + ".log"),
            os.path.join(projectDir, "runs", args.project_name + ".log"))


def slurmHeader(jobName, output, queue="shortq", ntasks=1, time="10:00:00",
                cpusPerTask=16, memPerCpu=2000, nodes=1, userMail=""):
    command = """    #!/bin/bash
    #SBATCH --partition={0}
    #SBATCH --ntasks={1}
    #SBATCH --time={2}

    #SBATCH --cpus-per-task={3}
    #SBATCH --mem-per-cpu={4}
    #SBATCH --nodes={5}

    #SBATCH --job-name={6}
    #SBATCH --output={7}

    #SBATCH --mail-type=end
    #SBATCH --mail-user={8}

    # Start running the job
    hostname
    date

    """.format(queue, ntasks, time, cpusPerTask, memPerCpu,
               nodes, jobName, output, userMail)

    return command


def slurmFooter():
    command = """
    # Job end
    date

    """

    return command


def slurmSubmitJob(jobFile):
    command = "sbatch %s" % jobFile

    return os.system(command)


def removeFile(fileName):
    command = """
    # Removing file
    rm {0}
    """.format(fileName)

    return command


def makeDir(directory):
    command = """
    # Removing file

    mkdir -p {0}
    """.format(directory)

    return command


def performWork(input, output):
    pass


def readImage(file):
    return = skimage.io.imread(file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Program canceled by user!")
        sys.exit(1)
