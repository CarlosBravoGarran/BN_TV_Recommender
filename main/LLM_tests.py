#!/usr/bin/env python3
"""
Test script for intent classification
Run this to verify intent classification is working correctly
"""

import sys
sys.path.insert(0, 'main')

from LLM_agent import classify_intent, colorize, INTENT_COLORS

# Test cases with expected intent
TEST_CASES = [
    # Genre rejections (should be ALTERNATIVE)
    ("No me gusta el drama", "ALTERNATIVE"),
    ("Nada de terror", "ALTERNATIVE"),
    ("No quiero ver comedias", "ALTERNATIVE"),
    ("Algo que no sea romántico", "ALTERNATIVE"),
    ("Prefiero otro género", "ALTERNATIVE"),
    
    # Content rejections (should be FEEDBACK_NEG)
    ("Esa no me gusta", "FEEDBACK_NEG"),
    ("Ya la vi", "FEEDBACK_NEG"),
    ("No me convence esa película", "FEEDBACK_NEG"),
    ("Esa es muy larga", "FEEDBACK_NEG"),
    
    # Simple alternatives (should be ALTERNATIVE)
    ("Dame otra opción", "ALTERNATIVE"),
    ("Otra cosa", "ALTERNATIVE"),
    ("Algo diferente", "ALTERNATIVE"),
    
    # Positive feedback (should be FEEDBACK_POS)
    ("Me gusta", "FEEDBACK_POS"),
    ("Perfecto", "FEEDBACK_POS"),
    ("La veo", "FEEDBACK_POS"),
    ("De acuerdo", "FEEDBACK_POS"),
    
    # Recommendations (should be RECOMMEND)
    ("Quiero ver una película", "RECOMMEND"),
    ("Qué puedo ver", "RECOMMEND"),
    ("Recomiéndame algo", "RECOMMEND"),
    
    # Smalltalk (should be SMALLTALK)
    ("Hola", "SMALLTALK"),
    ("Gracias", "SMALLTALK"),
    ("Adiós", "SMALLTALK"),
]


def test_intent_classification():
    """Test intent classification with predefined cases"""
    
    print("Testing Intent Classification\n")
    print("=" * 80)
    
    correct = 0
    total = len(TEST_CASES)
    errors = []
    
    for message, expected_intent in TEST_CASES:
        try:
            detected_intent = classify_intent(message)
            
            is_correct = detected_intent == expected_intent
            
            if is_correct:
                correct += 1
                status = colorize("✅ PASS", "\033[92m")
            else:
                status = colorize("❌ FAIL", "\033[91m")
                errors.append((message, expected_intent, detected_intent))
            
            print(f"{status} | {message:40} | Expected: {expected_intent:15} | Got: {detected_intent}")
            
        except Exception as e:
            print(f"❌ ERROR | {message:40} | {str(e)}")
            errors.append((message, expected_intent, f"ERROR: {e}"))
    
    print("=" * 80)
    print(f"\nResults: {correct}/{total} correct ({100*correct/total:.1f}%)")
    
    if errors:
        print(f"\n❌ {len(errors)} errors found:\n")
        for msg, expected, got in errors:
            print(f"  Message: '{msg}'")
            print(f"  Expected: {expected}")
            print(f"  Got: {got}")
            print()
    else:
        print("\n🎉 All tests passed!")
    
    return correct == total


if __name__ == "__main__":
    success = test_intent_classification()
    sys.exit(0 if success else 1)