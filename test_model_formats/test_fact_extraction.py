"""
Test fact extraction prompt without running full evaluation.

This script tests ONLY the fact extraction prompt (first step) by:
1. Reading a transcription from a file
2. Applying the custom reading comprehension prompt
3. Showing what facts are extracted

Usage:
    # Save your transcription to a file first
    echo "The United Methodist Church, UMC, is a mainline Protestant..." > transcription.txt
    
    # Test fact extraction
    python test_fact_extraction.py transcription.txt
    
    # Or with custom prompt file
    python test_fact_extraction.py transcription.txt --prompt custom_prompt.txt
"""

import argparse
import json
import os
import sys
from openai import OpenAI


# Default: Use the custom reading comprehension prompt
# You can override this with --prompt argument
DEFAULT_PROMPT_PATH = "../evaluation/audio_eval/prompts.py"


def load_custom_prompt():
    """Load the reading comprehension prompt from prompts.py"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../evaluation")
    from audio_eval.prompts import READING_COMPREHENSION_MEMORY_PROMPT
    return READING_COMPREHENSION_MEMORY_PROMPT


def extract_facts(transcription: str, prompt: str, api_key: str) -> dict:
    """
    Extract facts from transcription using the prompt.
    
    Args:
        transcription: The text to extract facts from
        prompt: The system prompt to use
        api_key: OpenAI API key
        
    Returns:
        Dict with "facts" key containing list of extracted facts
    """
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Input: user:\n{transcription}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    result = json.loads(response.choices[0].message.content)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test fact extraction prompt on a transcription"
    )
    parser.add_argument(
        "transcription_file",
        help="Path to file containing transcription text"
    )
    parser.add_argument(
        "--prompt",
        help="Path to custom prompt file (default: uses READING_COMPREHENSION_MEMORY_PROMPT)",
        default=None
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: from OPENAI_API_KEY env var)",
        default=os.getenv("OPENAI_API_KEY")
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not args.api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    # Load transcription
    if not os.path.exists(args.transcription_file):
        print(f"❌ Error: File not found: {args.transcription_file}")
        sys.exit(1)
    
    with open(args.transcription_file, 'r') as f:
        transcription = f.read().strip()
    
    if not transcription:
        print("❌ Error: Transcription file is empty")
        sys.exit(1)
    
    print("=" * 70)
    print("FACT EXTRACTION TEST")
    print("=" * 70)
    
    # Load prompt
    if args.prompt:
        print(f"\n📝 Loading custom prompt from: {args.prompt}")
        with open(args.prompt, 'r') as f:
            prompt = f.read().strip()
    else:
        print(f"\n📝 Using default: READING_COMPREHENSION_MEMORY_PROMPT")
        prompt = load_custom_prompt()
    
    print(f"\n📄 Transcription ({len(transcription)} chars):")
    print("-" * 70)
    print(transcription[:500] + "..." if len(transcription) > 500 else transcription)
    print("-" * 70)
    
    # Extract facts
    print("\n🔄 Extracting facts using GPT-4o-mini...")
    
    try:
        result = extract_facts(transcription, prompt, args.api_key)
        facts = result.get("facts", [])
        
        print("\n✅ Extraction complete!")
        print("=" * 70)
        print(f"EXTRACTED FACTS ({len(facts)} total)")
        print("=" * 70)
        
        if facts:
            for i, fact in enumerate(facts, 1):
                print(f"\n{i}. {fact}")
        else:
            print("\n(No facts extracted)")
        
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        # Check for definitional facts
        definitional = [f for f in facts if "stands for" in f.lower() or "means" in f.lower()]
        numerical = [f for f in facts if any(char.isdigit() for char in f)]
        
        print(f"\nTotal facts: {len(facts)}")
        print(f"Definitional facts (stands for, means): {len(definitional)}")
        print(f"Facts with numbers/dates: {len(numerical)}")
        print(f"Average fact length: {sum(len(f) for f in facts) / len(facts):.1f} chars" if facts else "N/A")
        
        # Check if under 30 facts (consolidation target)
        if len(facts) > 30:
            print(f"\n⚠️  Warning: {len(facts)} facts extracted (target: ≤30)")
            print("   The prompt should consolidate to stay under 30 facts")
        else:
            print(f"\n✓ Good: {len(facts)} facts (under consolidation target of 30)")
        
        print("\n" + "=" * 70)
        
        # Save results
        output_file = args.transcription_file.replace('.txt', '_facts.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

