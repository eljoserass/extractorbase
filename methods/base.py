# define the interfaces protocosl of a extractor
# something like fit() and it has some basic params, the tricky part is the input can be different for each. 
# the output should be somewhat the same, maybe each can have different requirements and stuff
# but basically all should somewhat fit() and predict() based on its input and output, and have a way to load and dump its config
# - we will later define something prettier to load and dumpt that its a json/yaml and not a nasty pickle