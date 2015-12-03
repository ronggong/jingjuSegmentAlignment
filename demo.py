import numpy as np
import os,sys

# add src patt
dir = os.path.dirname(os.path.realpath(__file__))
srcpath = dir+'/src'
sys.path.append(srcpath)

import concatenateSegment as cs
import dtwRong
import dtwSankalp
import plottingCode as pc
import alignment as align

# definition
# segmentFileFolder = './segmentFiles/'
# outputFileFolder = './outputFiles/'

mp3Folder = '/Users/gong/Documents/github/MTG/ICASSP2016/exampleAudios/07/'
segmentFileFolder = mp3Folder
outputFileFolder = mp3Folder

teacherTitle = 'teacher'
studentTitle = 'student'

teacherMonoNoteOutFilename = segmentFileFolder+teacherTitle+'_monoNoteOut_midi.csv'
studentMonoNoteOutFilename = segmentFileFolder+studentTitle+'_monoNoteOut_midi.csv'

teacherRepresentationFilename = segmentFileFolder+teacherTitle+'_representation.json'
studentRepresentationFilename = segmentFileFolder+studentTitle+'_representation.json'

teacherNoteAlignedFilename = outputFileFolder+teacherTitle+'_noteAligned.csv'
studentNoteAlignedFilename = outputFileFolder+studentTitle+'_noteAligned.csv'

teacherSegAlignedFilename = outputFileFolder+teacherTitle+'_segAligned.csv'
studentSegAlignedFilename = outputFileFolder+studentTitle+'_segAligned.csv'

cs1 = cs.ConcatenateSegment()
align1 = align.Alignment()

#################################################### note alignment ####################################################

# read note file
noteStartingTime_t, noteDurTime_t, midiNote_t = cs1.readPyinMonoNoteOutMidi(teacherMonoNoteOutFilename)
noteStartingTime_s, noteDurTime_s, midiNote_s = cs1.readPyinMonoNoteOutMidi(studentMonoNoteOutFilename)
noteStartingFrame_t, noteEndingFrame_t = cs1.getNoteFrameBoundary(noteStartingTime_t, noteDurTime_t)
noteStartingFrame_s, noteEndingFrame_s = cs1.getNoteFrameBoundary(noteStartingTime_s, noteDurTime_s)

# get concatenated pitch track
notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t = \
    cs1.generateNotePitchtrack(noteStartingFrame_t, noteEndingFrame_t, midiNote_t)
notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s = \
    cs1.generateNotePitchtrack(noteStartingFrame_s, noteEndingFrame_s, midiNote_s)

# normalization pitch track
notePts_t = cs1.pitchtrackNormalization(notePts_t)
notePts_s = cs1.pitchtrackNormalization(notePts_s)

# resampling note
if len(notePts_t) > len(notePts_s):
    notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s \
        = cs1.resampling(notePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s, len(notePts_t))
elif len(notePts_s) > len(notePts_t):
    notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t \
        = cs1.resampling(notePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t, len(notePts_s))

# alignment
path = dtwSankalp.dtw1d_generic(notePts_t,notePts_s)
path_t = path[0]
path_s = path[1]

# get path index for each note
noteStartingFramePath_t, noteEndingFramePath_t \
    = align1.getPathIndex(path_t,noteStartingFrameConcatenate_t,noteEndingFrameConcatenate_t)
noteStartingFramePath_s, noteEndingFramePath_s \
    = align1.getPathIndex(path_s,noteStartingFrameConcatenate_s,noteEndingFrameConcatenate_s)

# alignment
alignedNote_t = align1.alignment2(noteStartingFramePath_t, noteEndingFramePath_t,
                            noteStartingFramePath_s, noteEndingFramePath_s)

alignedNote_s = align1.alignment2(noteStartingFramePath_s, noteEndingFramePath_s,
                            noteStartingFramePath_t, noteEndingFramePath_t)

# print noteStartingFramePath_t, noteEndingFramePath_t
# print noteStartingFramePath_s, noteEndingFramePath_s

# print alignedNote_t, alignedNote_s


############################################ segmentation alignment ####################################################

segmentPts_t,boundaries_t,target_t = cs1.readRepresentation(teacherRepresentationFilename)
segmentPts_s,boundaries_s,target_s = cs1.readRepresentation(studentRepresentationFilename)

concatenatePts_t,segStartingFrame_t,segEndingFrame_t = cs1.concatenate(segmentPts_t)
concatenatePts_s,segStartingFrame_s,segEndingFrame_s = cs1.concatenate(segmentPts_s)

concatenatePts_t = cs1.pitchtrackNormalization(concatenatePts_t)
concatenatePts_s = cs1.pitchtrackNormalization(concatenatePts_s)

# print len(segmentPts_t), len(segmentPts_s)
# print segStartingFrame_t, segStartingFrame_s

# resampling note
if len(notePts_t) > len(notePts_s):
    concatenatePts_s, noteStartingFrameConcatenate_s, noteEndingFrameConcatenate_s \
        = cs1.resampling(concatenatePts_s, segStartingFrame_s, segEndingFrame_s, len(concatenatePts_t))
elif len(notePts_s) > len(notePts_t):
    concatenatePts_t, noteStartingFrameConcatenate_t, noteEndingFrameConcatenate_t \
        = cs1.resampling(concatenatePts_t, segStartingFrame_t, segEndingFrame_t, len(concatenatePts_s))

# do dtw
#dist, D, path = dtwRong.dtw(concatenatePts_t,concatenatePts_s)
path = dtwSankalp.dtw1d_generic(concatenatePts_t,concatenatePts_s)
path_t = path[0]
path_s = path[1]

# print path_t, path_s

# get path index for each note
segStartingFramePath_t, segEndingFramePath_t \
    = align1.getPathIndex(path_t,segStartingFrame_t,segEndingFrame_t)
segStartingFramePath_s, segEndingFramePath_s \
    = align1.getPathIndex(path_s,segStartingFrame_s,segEndingFrame_s)

# alignment
alignedSeg_t = align1.alignment2(segStartingFramePath_t, segEndingFramePath_t,
                                segStartingFramePath_s, segEndingFramePath_s)

alignedSeg_s = align1.alignment2(segStartingFramePath_s, segEndingFramePath_s,
                                segStartingFramePath_t, segEndingFramePath_t)

# print alignedSeg_t, alignedSeg_s

############################################ save aligned file #########################################################

def stral(al):
    out = ''
    for ii, c in enumerate(al):
        if ii != len(al)-1:
            out = out+str(c)+' '
        else:
            out = out+str(c)
    return out

def writeAlignedFile(alignedFilename, alignedResult):
    with open(alignedFilename, 'w+') as outfile:
        outfile.write('teacher'+','+'student'+'\n')
        for al in alignedResult:
            if not al[1]:
                alignStr = 'null'
            else:
                alignStr = stral(al[1])
            outfile.write(str(al[0])+','+alignStr+'\n')

writeAlignedFile(teacherNoteAlignedFilename, alignedNote_t)
writeAlignedFile(studentNoteAlignedFilename, alignedNote_s)
writeAlignedFile(teacherSegAlignedFilename, alignedSeg_t)
writeAlignedFile(studentSegAlignedFilename, alignedSeg_s)
