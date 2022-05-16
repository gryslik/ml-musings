import eagle_large
import sys

if len(sys.argv) == 1:
    eagle_large.train_agent()
elif len(sys.argv) == 2:
    eagle_large.record_model(sys.argv[1])

