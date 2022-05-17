import eagle_large
import sys

# If no arguments are passed, train the agent
if len(sys.argv) == 1:
    eagle_large.train_agent()
# If an arguement is passed, assume it is a model and record it
elif len(sys.argv) == 2:
    eagle_large.record_model(sys.argv[1])

