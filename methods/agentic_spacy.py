# you pass the samples, all of them can be used for training maybe bc evaluator hides the test (TODO) maybe check actually
#                                                                                            how other libraries do it
#                                                           who handles the responsability? the evaluateor "hides" the test? 
# we call an agent (this will be done later by an external module bc we might use the provisioning for many things, for now just do something inside here that its easy to replace later)
# you give the agent the input (which is the samples + some kind of prompts that the doctor gave)
# (we assume the agent is already prepared for this, thats why agent provisiioning useufl, just call the type of agent you want)
# then the agent will start to creating the spacy rules, execute them, test it on the data, iterating improving result
# - it should be possible, maybe in another phase to have the option to iterate with the evaluator, it hinding the data form the agent,
#       communicating its result on the test set, maybe some another hints, and putting the number of rounds to keep trying
# - also to iterate with the user, so able to communicate its process, and chat to keep improving
# from the beggining iterate witha capable agent taht is configurable and has reasonable license, so we could do pi agent 