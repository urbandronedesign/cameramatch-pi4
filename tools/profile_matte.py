"""Profile one fused matte run: python /tools/profile_matte.py /work/<job>"""
import cProfile
import pstats
import sys

import matte_pipeline as mp

prof = cProfile.Profile()
prof.enable()
mp.run(sys.argv[1], "fused", write=["fused"], crops=False, occlusion=True)
prof.disable()
pstats.Stats(prof).sort_stats("cumulative").print_stats(30)
