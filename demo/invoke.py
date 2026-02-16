import subprocess
from typing import Optional, Any, Type
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_aws import ChatBedrockConverse
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI 
import tracking_aws
import requests
import time
from botocore.exceptions import ClientError
from config import Config
from langchain_ollama import ChatOllama
import random
import os

class LLMService:
    def __init__(self, config: object):
        # Default to a gemini model if provider is gemini but no version specified
        self.model_version = getattr(config, "model_version")
        self.temperature = getattr(config, "temperature")
        self.model_provider = getattr(config, "model_provider")
        
        # Initialize statistics
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.failed_calls = 0
        self.retry_count = 0
        
        # Initialize the LLM
        if self.model_provider.lower() == "bedrock":
            bedrock_runtime = tracking_aws.new_default_client()
            self.llm = ChatBedrockConverse(
                client=bedrock_runtime, 
                model_id=self.model_version, 
                temperature=self.temperature, 
                max_tokens=8192
            )
        elif self.model_provider.lower() == "anthropic":
            self.llm = ChatAnthropic(
                model=self.model_version, 
                temperature=self.temperature
            )
        elif self.model_provider.lower() == "openai":
            self.llm = init_chat_model(
                self.model_version, 
                model_provider=self.model_provider, 
                temperature=self.temperature
            )
        elif self.model_provider.lower() == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_version,
                temperature=self.temperature,
                max_output_tokens=8192,
                api_key=os.environ.get("GEMINI_API_KEY")
            )
        elif self.model_provider.lower() == "ollama":
            try:
                response = requests.get("http://localhost:11434/api/version", timeout=2)
            except requests.exceptions.RequestException:
                print("Ollama is not running, starting it...")
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                time.sleep(5)

            self.llm = ChatOllama(
                model=self.model_version, 
                temperature=self.temperature,
                num_predict=-1,
                num_ctx=131072,
                base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"{self.model_provider} is not a supported model_provider")
    
    def invoke(self, 
               user_prompt: str, 
               system_prompt: Optional[str] = None, 
               pydantic_obj: Optional[Type[BaseModel]] = None,
               max_retries: int = 10) -> Any:
        """
        Invoke the LLM with the given prompts and return the response.
        """
        self.total_calls += 1
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Calculate prompt tokens (approximate for raw text)
        prompt_tokens = 0
        # Simple token estimation if get_num_tokens fails or for initial guess
        try:
            for message in messages:
                prompt_tokens += self.llm.get_num_tokens(str(message["content"]))
        except:
            pass
        
        retry_count = 0
        while True:
            try:
                if pydantic_obj:
                    structured_llm = self.llm.with_structured_output(pydantic_obj)
                    response = structured_llm.invoke(messages)
                else:
                    response = self.llm.invoke(messages)
                    response = response.content
                return response
                
            except ClientError as e:
                if e.response['Error']['Code'] in ['Throttling', 'TooManyRequestsException']:
                    retry_count += 1
                    self.retry_count += 1
                    
                    if retry_count > max_retries:
                        self.failed_calls += 1
                        raise Exception(f"Maximum retries ({max_retries}) exceeded: {str(e)}")
                    
                    base_delay = 1.0
                    max_delay = 60.0
                    delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter
                    
                    print(f"ThrottlingException: {str(e)}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    self.failed_calls += 1
                    raise e
            except Exception as e:
                self.failed_calls += 1
                raise e

prompt_writer = LLMService( Config() )