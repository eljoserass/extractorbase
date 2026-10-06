generalized structure of text, for medical corporas mostly

the system will evolve into an user facing platform, which they can define the structure, give some examples
and "teach" the extractor to work on those samples with a structure defined

it is a priority that it should run on local, an idea is that the extractors defined perharps the initial fitting, there might be some that work on the cloud, assuming the sent initial samples do not contain any 
personal data, but then the result can be ran locally (for instance, a capable coding model iterates on generating code for spacy rules, which then can be easilly run locally, or iterating on the prompt of a smaller llm, with a bigger llm teacher (see methods/ agentic spacy and llm for some definitions of this))

thats why its needed that the core should be somwhat extensible to something different that the cli

the v1 would be approximately:
- download a desktop app (or access a website TBD)
- create a project
- interface to define/upload desired extrucure
- iteration on the extractor, upload some examples that match the format input:text -> output:the defined schema
- by default a extractor is selected but user should be able to select the desired one
- the extractor starts "learning" from the exmamples as a parallel job
- user notified when job is finished, visualize results, and errors. extractor not yet finished until the user signs, in this process user can change samples/structure/prompts/ highhlight erorrs etc.
- user sign when is comfortable with the result. client receives the "trained" model.
- user can access the trained extractor, and pass more (labelled or unlabled) samples that will run locally in parallel.
- user can see a result of the run, export results etc


the v0.1 will look like:
- set some of the main interfaces that could be resued later. for now:
    - a way to load data, handle errors different structures etc
    - an evaluator that can handle the input ouput, computing the metrics, training splits etc.
    - interoperable base between extractors
    - main basic extractors evaluated
    - the demostration will be done with a cli calling the main components, on the reumalago anotated dataset
    - for now only a python cli, no distinctino between user facing application or whatever