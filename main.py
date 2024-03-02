import openai
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain,SequentialChain
from langchain.agents import AgentType,Agent,Tool, initialize_agent,load_tools
from langchain.memory import ConversationBufferMemory
#cities = ['bangalore', 'punjab', 'bangalore', 'pune']
#llm = OpenAI(temperature=0.3)
#prompt = PromptTemplate.from_template("What is the capital of {city}?")
#chain = LLMChain(llm=llm, prompt=prompt)
#for city in cities:
 # output = chain.run(city)
#print(output)

#prompt.format(city="New York")
#print(llm.predict())

#LLM to get name of an e commerce store from a product name
prompt=PromptTemplate.from_template("What is the name of the e commerce store that sells {product}?")
llm=OpenAI(temperature=0.3)
chain1=LLMChain(llm=llm, prompt=prompt)
#product="iphone"
#output=chain.run(product)
#print(output)

#llm to get comma seperted name of products from e commerce store 
prompt=PromptTemplate.from_template("What is the name of products at {store}?")
llm=OpenAI(temperature=0.3)
chain2=LLMChain(llm=llm, prompt=prompt)
#store="Amazon"
#output=chain.run(store)
#print(output)

#create overall chain from simple sequential chain
chain=SimpleSequentialChain(chains=[chain1, chain2],verbose=True)
chain.run("candles")

#write a synopsis given a title of play,see everything in output
llm=OpenAI(temperature=0.7)
template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.
Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

#review of play synopsis
llm=OpenAI(temperature=0.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

overall_chain = SequentialChain(chains=[synopsis_chain, review_chain], input_variables=["era", "title"],output_variables=["synopsis", "review"], verbose=True)

#print(overall_chain({"era": "Renaissance", "title": "The Tempest"}))

#generate agent
llm=OpenAI(temperature=0.7)
tools=load_tools(["wikipedia","llm-math"],llm=llm)
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("How old is Varun Dhawan in 2023?")

#memory 
llm = OpenAI(temperature=0.3)
prompt = PromptTemplate.from_template("What is the name of the e commerce store that sells {product}?")
chain = LLMChain(llm=llm, prompt=prompt, memory=ConversationBufferMemory())
output = chain.run("fruits")
output = chain.run("books") 
print(chain.memory.buffer)
print(output)