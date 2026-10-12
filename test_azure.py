import asyncio
import os
import sys
from dotenv import load_dotenv

# Load .env
load_dotenv()


async def test_azure_openai():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-4o-mini"
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    print("=" * 60)
    print(" Azure OpenAI Diagnostics & Test Script")
    print("=" * 60)
    print(f"Endpoint:        {endpoint}")
    print(f"Deployment Name: {deployment_name}")
    print(f"API Version:     {api_version}")
    print(f"API Key:         {'*' * 8 + api_key[-4:] if api_key else 'NOT SET'}")
    print("-" * 60)

    if not endpoint or not api_key:
        print("[ERROR] AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY is missing in .env")
        return

    # Method 1: Testing with LangChain AzureChatOpenAI (used by NotionAgent)
    print("\n[1/2] Testing LangChain `AzureChatOpenAI`...")
    try:
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment_name,
            api_key=api_key,
            api_version=api_version,
            temperature=0.3,
        )

        test_prompt = "Respond with: 'Azure OpenAI is connected and working successfully!'"
        print(f"Sending prompt: \"{test_prompt}\"")
        
        response = await llm.ainvoke(test_prompt)
        print("\n>>> LangChain Response Received:")
        print(response.content)
        print("\n[SUCCESS] LangChain AzureChatOpenAI works properly!")

    except Exception as e:
        print(f"\n[FAILED] LangChain test failed with {type(e).__name__}:")
        print(f"Details: {e}")
        _print_troubleshooting_tips(e, deployment_name)
        return

    # Method 2: Testing with native OpenAI AzureOpenAI client
    print("\n" + "-" * 60)
    print("[2/2] Testing native `openai.AsyncAzureOpenAI` client...")
    try:
        from openai import AsyncAzureOpenAI

        native_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        chat_completion = await native_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Give a 1-sentence confirmation."},
            ],
            max_tokens=50,
        )
        print("\n>>> Native OpenAI Response Received:")
        print(chat_completion.choices[0].message.content)
        print("\n[SUCCESS] Native AsyncAzureOpenAI works properly!")

    except Exception as e:
        print(f"\n[FAILED] Native client test failed with {type(e).__name__}: {e}")
        _print_troubleshooting_tips(e, deployment_name)


def _print_troubleshooting_tips(exc: Exception, deployment_name: str):
    error_msg = str(exc)
    print("\n" + "=" * 60)
    print("Troubleshooting Guidance:")
    print("=" * 60)
    if "DeploymentNotFound" in error_msg or "404" in error_msg:
        print(f"Reason: The deployment '{deployment_name}' was NOT found on your Azure OpenAI resource.")
        print("\nHow to fix:")
        print("1. Open the Azure AI Foundry / Azure OpenAI Studio portal:")
        print("   https://oai.azure.com/ or https://ai.azure.com/")
        print("2. Navigate to your resource ('sivaopenai123') -> 'Deployments' tab.")
        print("3. Check the exact string in the 'Deployment name' column.")
        print("   (Note: The 'Deployment name' you chose when deploying might differ from the underlying base model name).")
        print(f"4. Update your .env file with that exact name:")
        print(f"   AZURE_OPENAI_DEPLOYMENT_NAME=<exact_deployment_name>")
    elif "401" in error_msg or "AccessDenied" in error_msg:
        print("Reason: Invalid API key or unauthorized access.")
        print("Check AZURE_OPENAI_API_KEY in your .env file.")
    else:
        print("Check your network, endpoint URL, or Azure API version.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_azure_openai())
