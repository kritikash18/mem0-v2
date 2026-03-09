"""
Quick sanity check for Ollama setup.

Verifies that Ollama is installed, running, and can generate responses
with llama3.2 before attempting a full evaluation run.

Usage:
    # Step 1: Install Ollama and pull the model (do this first in terminal)
    #   brew install ollama        (or download from ollama.com)
    #   ollama serve               (start the server — leave running)
    #   ollama pull llama3.2       (download the model ~2GB)

    # Step 2: Run this script
    python test_model_formats/test_ollama_setup.py

    # Step 3 (optional): Test with a specific model
    python test_model_formats/test_ollama_setup.py --model llama3.1:8b
"""

import argparse
import json
import sys
import time


def check_ollama_installed():
    """Check if the ollama Python library is installed."""
    print("=" * 60)
    print("CHECK 1: Python 'ollama' library")
    print("=" * 60)
    try:
        import ollama
        print(f"  ✓ ollama library found (version: {getattr(ollama, '__version__', 'unknown')})")
        return True
    except ImportError:
        print("  ✗ ollama library NOT installed")
        print("  Fix: pip install ollama")
        return False


def check_server_running(base_url="http://localhost:11434"):
    """Check if the Ollama server is reachable."""
    print()
    print("=" * 60)
    print("CHECK 2: Ollama server")
    print("=" * 60)
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"  ✓ Server running at {base_url}")
            print(f"  ✓ {len(models)} model(s) available:")
            for m in models:
                size_gb = m.get("size", 0) / (1024 ** 3)
                print(f"      - {m['name']} ({size_gb:.1f} GB)")
            return True, [m["name"] for m in models]
        else:
            print(f"  ✗ Server responded with status {resp.status_code}")
            return False, []
    except requests.ConnectionError:
        print(f"  ✗ Cannot connect to {base_url}")
        print("  Fix: Open a new terminal and run: ollama serve")
        return False, []
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, []


def check_model_available(model_name, available_models):
    """Check if the desired model is downloaded."""
    print()
    print("=" * 60)
    print(f"CHECK 3: Model '{model_name}'")
    print("=" * 60)

    # Ollama model names may include :latest tag
    matches = [m for m in available_models if m.startswith(model_name.split(":")[0])]

    if matches:
        print(f"  ✓ Model found: {matches[0]}")
        return True
    else:
        print(f"  ✗ Model '{model_name}' not downloaded")
        print(f"  Fix: ollama pull {model_name}")
        return False


def test_basic_chat(model_name, base_url="http://localhost:11434"):
    """Send a simple chat message and verify the response."""
    print()
    print("=" * 60)
    print(f"CHECK 4: Basic chat with '{model_name}'")
    print("=" * 60)

    from ollama import Client

    client = Client(host=base_url)

    messages = [
        {"role": "user", "content": "Reply with exactly one word: Hello"}
    ]

    print(f"  Sending: '{messages[0]['content']}'")
    start = time.time()

    try:
        response = client.chat(model=model_name, messages=messages)
        elapsed = time.time() - start
        content = response.message.content if hasattr(response, "message") else response["message"]["content"]
        print(f"  Response: '{content.strip()}'")
        print(f"  Time: {elapsed:.2f}s")
        print("  ✓ Basic chat works!")
        return True
    except Exception as e:
        print(f"  ✗ Chat failed: {e}")
        return False


def test_json_output(model_name, base_url="http://localhost:11434"):
    """Test JSON-structured output (used by mem0 for fact extraction)."""
    print()
    print("=" * 60)
    print(f"CHECK 5: JSON output (fact extraction format)")
    print("=" * 60)

    from ollama import Client

    client = Client(host=base_url)

    messages = [
        {
            "role": "system",
            "content": "You extract facts from text. Return JSON with a 'facts' key containing a list of strings."
        },
        {
            "role": "user",
            "content": 'Mount Everest is 8,848 meters tall and is located in Nepal.\n\nPlease respond with valid JSON only.'
        }
    ]

    print("  Sending fact extraction request...")
    start = time.time()

    try:
        response = client.chat(model=model_name, messages=messages, format="json")
        elapsed = time.time() - start
        content = response.message.content if hasattr(response, "message") else response["message"]["content"]

        print(f"  Raw response: {content.strip()[:200]}")
        print(f"  Time: {elapsed:.2f}s")

        parsed = json.loads(content)
        facts = parsed.get("facts", [])
        print(f"  Parsed {len(facts)} fact(s):")
        for f in facts:
            print(f"    - {f}")
        print("  ✓ JSON output works!")
        return True
    except json.JSONDecodeError as e:
        print(f"  ✗ Response is not valid JSON: {e}")
        print("  This may cause issues with mem0 fact extraction.")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_mem0_integration(model_name, base_url="http://localhost:11434"):
    """Test that mem0 can use Ollama as its LLM provider."""
    print()
    print("=" * 60)
    print(f"CHECK 6: mem0 integration")
    print("=" * 60)

    try:
        from mem0.llms.ollama import OllamaLLM
        from mem0.configs.llms.ollama import OllamaConfig

        config = OllamaConfig(
            model=model_name,
            temperature=0.0,
            ollama_base_url=base_url,
        )
        llm = OllamaLLM(config)

        print(f"  ✓ OllamaLLM created with model={model_name}")

        messages = [{"role": "user", "content": "Say 'OK' and nothing else."}]
        start = time.time()
        response = llm.generate_response(messages=messages)
        elapsed = time.time() - start

        print(f"  Response: '{response.strip()}'")
        print(f"  Time: {elapsed:.2f}s")
        print("  ✓ mem0 OllamaLLM integration works!")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ mem0 integration failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Ollama setup for evaluation")
    parser.add_argument(
        "--model", type=str, default="llama3.2",
        help="Ollama model to test (default: llama3.2)",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:11434",
        help="Ollama server URL",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         OLLAMA SETUP VERIFICATION                   ║")
    print(f"║  Model: {args.model:<44}║")
    print(f"║  Server: {args.base_url:<43}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    results = {}

    # Check 1: Library installed
    results["library"] = check_ollama_installed()
    if not results["library"]:
        print("\n❌ Cannot proceed without the ollama library. Run: pip install ollama")
        sys.exit(1)

    # Check 2: Server running
    results["server"], available = check_server_running(args.base_url)
    if not results["server"]:
        print("\n❌ Ollama server not running. Open a new terminal and run: ollama serve")
        sys.exit(1)

    # Check 3: Model available
    results["model"] = check_model_available(args.model, available)
    if not results["model"]:
        print(f"\n❌ Model not downloaded. Run: ollama pull {args.model}")
        sys.exit(1)

    # Check 4: Basic chat
    results["chat"] = test_basic_chat(args.model, args.base_url)

    # Check 5: JSON output
    results["json"] = test_json_output(args.model, args.base_url)

    # Check 6: mem0 integration
    results["mem0"] = test_mem0_integration(args.model, args.base_url)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = all(results.values())
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    if all_pass:
        print(f"\n✅ All checks passed! You can run the evaluation with:")
        print(f"   python -m audio_eval.evaluator -n 10 \\")
        print(f"     --llm-provider ollama --llm-model {args.model} \\")
        print(f"     -e ollama_test")
    else:
        print("\n⚠️  Some checks failed. Fix the issues above before running evaluation.")

    print()


if __name__ == "__main__":
    main()
