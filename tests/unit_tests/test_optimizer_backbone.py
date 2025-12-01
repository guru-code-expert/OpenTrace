"""
Comprehensive tests for optimizer backbone components (ConversationHistory, UserTurn, AssistantTurn)
Tests include: truncation strategies, multimodal content, and conversation management
"""
import pytest
import base64
from opto.optimizers.backbone import (
    ConversationHistory,
    UserTurn,
    AssistantTurn,
    TextContent,
    ImageContent
)


# ============================================================================
# Test Fixtures
# ============================================================================

def create_sample_conversation():
    """Create a sample conversation with multiple rounds"""
    history = ConversationHistory(system_prompt="You are a helpful assistant.")
    
    # Round 1
    user1 = UserTurn().add_text("Hello, what's the weather?")
    assistant1 = AssistantTurn().add_text("The weather is sunny today.")
    history.add_user_turn(user1).add_assistant_turn(assistant1)
    
    # Round 2
    user2 = UserTurn().add_text("What about tomorrow?")
    assistant2 = AssistantTurn().add_text("Tomorrow will be rainy.")
    history.add_user_turn(user2).add_assistant_turn(assistant2)
    
    # Round 3
    user3 = UserTurn().add_text("Should I bring an umbrella?")
    assistant3 = AssistantTurn().add_text("Yes, definitely bring an umbrella.")
    history.add_user_turn(user3).add_assistant_turn(assistant3)
    
    # Round 4
    user4 = UserTurn().add_text("Thanks for the advice!")
    assistant4 = AssistantTurn().add_text("You're welcome! Stay dry!")
    history.add_user_turn(user4).add_assistant_turn(assistant4)
    
    return history


# ============================================================================
# Truncation Tests
# ============================================================================

def test_default_all_history():
    """Test default behavior (n=-1) returns all history"""
    history = create_sample_conversation()
    
    messages = history.to_messages()
    
    # Should have: system + 8 turns (4 user + 4 assistant)
    assert len(messages) == 9  # 1 system + 8 messages
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[-1]["role"] == "assistant"


def test_truncate_from_start():
    """Test truncate_from_start strategy - keeps last N turns"""
    history = create_sample_conversation()
    
    # Keep last 3 turns
    messages = history.to_messages(n=3, truncate_strategy="from_start")
    
    # Should have: system + 3 turns
    assert len(messages) == 4  # 1 system + 3 messages
    assert messages[0]["role"] == "system"
    
    # Should have the last 3 turns
    # Last 3 turns are: assistant3, user4, assistant4
    assert messages[1]["role"] == "assistant"
    assert "umbrella" in messages[1]["content"]
    assert messages[2]["role"] == "user"
    assert "Thanks" in messages[2]["content"][0]["text"]
    assert messages[3]["role"] == "assistant"
    assert "welcome" in messages[3]["content"]


def test_truncate_from_end():
    """Test truncate_from_end strategy - keeps first N turns"""
    history = create_sample_conversation()
    
    # Keep first 3 turns
    messages = history.to_messages(n=3, truncate_strategy="from_end")
    
    # Should have: system + 3 turns
    assert len(messages) == 4  # 1 system + 3 messages
    assert messages[0]["role"] == "system"
    
    # Should have the first 3 turns
    # First 3 turns are: user1, assistant1, user2
    assert messages[1]["role"] == "user"
    assert "Hello" in messages[1]["content"][0]["text"]
    assert messages[2]["role"] == "assistant"
    assert "sunny" in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert "tomorrow" in messages[3]["content"][0]["text"]


def test_truncate_zero_turns():
    """Test truncating to 0 turns"""
    history = create_sample_conversation()
    
    messages = history.to_messages(n=0, truncate_strategy="from_start")
    
    # Should only have system message
    assert len(messages) == 1
    assert messages[0]["role"] == "system"


def test_truncate_more_than_available():
    """Test requesting more turns than available"""
    history = create_sample_conversation()
    
    # Request 100 turns but only have 8
    messages = history.to_messages(n=100, truncate_strategy="from_start")
    
    # Should return all available
    assert len(messages) == 9  # 1 system + 8 messages


def test_empty_conversation():
    """Test truncation on empty conversation"""
    history = ConversationHistory(system_prompt="Test")
    
    messages = history.to_messages(n=5)
    
    assert len(messages) == 1  # Just system
    assert messages[0]["role"] == "system"


def test_to_litellm_format_with_truncation():
    """Test to_litellm_format() also supports truncation"""
    history = create_sample_conversation()
    
    messages = history.to_litellm_format(n=2, truncate_strategy="from_end")
    
    # Should have: system + 2 turns
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_invalid_strategy():
    """Test that invalid strategy raises error"""
    history = create_sample_conversation()
    
    with pytest.raises(ValueError, match="Unknown truncate_strategy"):
        history.to_messages(n=2, truncate_strategy="invalid_strategy")


def test_negative_n_values():
    """Test that n=-1 returns all history"""
    history = create_sample_conversation()
    
    # n=-1 should return all
    messages_all = history.to_messages(n=-1)
    assert len(messages_all) == 9
    
    # Verify it's the same as not passing n at all
    messages_default = history.to_messages()
    assert len(messages_all) == len(messages_default)


# ============================================================================
# Multimodal / Multi-Image Tests
# ============================================================================

def test_user_turn_multiple_images():
    """Test that a user turn can have multiple images"""
    history = ConversationHistory()
    
    # Create a user turn with text and multiple images (like the OpenAI example)
    user_turn = (UserTurn()
                 .add_text("What are in these images? Is there any difference between them?")
                 .add_image(url="https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg")
                 .add_image(url="https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"))
    
    history.add_user_turn(user_turn)
    
    # Convert to LiteLLM format
    messages = history.to_litellm_format()
    
    # Should have 1 message
    assert len(messages) == 1
    
    user_msg = messages[0]
    assert user_msg["role"] == "user"
    
    # Content should be a list with 3 items: 1 text + 2 images
    assert len(user_msg["content"]) == 3
    
    # Check first item is text
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][0]["text"] == "What are in these images? Is there any difference between them?"
    
    # Check second item is first image
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][1]["image_url"]["url"] == "https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg"
    
    # Check third item is second image
    assert user_msg["content"][2]["type"] == "image_url"
    assert user_msg["content"][2]["image_url"]["url"] == "https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt134818d279038650/6668df6434f6fb5cd48aac34/beautiful-flowers-rose.jpeg"


def test_assistant_turn_multiple_images():
    """Test that an assistant turn can also have multiple images (for models that generate images)"""
    history = ConversationHistory()
    
    # Assistant turn with text and multiple images
    assistant_turn = (AssistantTurn()
                      .add_text("Here are two generated images based on your request:")
                      .add_image(url="https://example.com/generated1.png")
                      .add_image(url="https://example.com/generated2.png"))
    
    history.add_assistant_turn(assistant_turn)
    
    # Convert to LiteLLM format
    messages = history.to_litellm_format()
    
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    
    # Assistant should have text content
    assert "Here are two generated images" in messages[0]["content"]


def test_mixed_content_types_in_turn():
    """Test mixing text, images, and other content types in a single turn"""
    history = ConversationHistory()
    
    # Create a complex turn with multiple content types
    user_turn = (UserTurn()
                 .add_text("Please analyze these images and this document:")
                 .add_image(url="https://example.com/chart1.png")
                 .add_image(url="https://example.com/chart2.png")
                 .add_text("What patterns do you see?"))
    
    history.add_user_turn(user_turn)
    
    messages = history.to_litellm_format()
    
    assert len(messages) == 1
    user_msg = messages[0]
    
    # Should have 4 content blocks: text, image, image, text
    assert len(user_msg["content"]) == 4
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][2]["type"] == "image_url"
    assert user_msg["content"][3]["type"] == "text"


def test_multiple_images_with_base64():
    """Test multiple images using base64 encoding"""
    history = ConversationHistory()
    
    # Create fake base64 image data
    fake_image_data1 = base64.b64encode(b"fake image 1").decode('utf-8')
    fake_image_data2 = base64.b64encode(b"fake image 2").decode('utf-8')
    
    user_turn = (UserTurn()
                 .add_text("Compare these two images:")
                 .add_image(data=fake_image_data1, media_type="image/png")
                 .add_image(data=fake_image_data2, media_type="image/jpeg"))
    
    history.add_user_turn(user_turn)
    
    messages = history.to_litellm_format()
    
    assert len(messages) == 1
    user_msg = messages[0]
    
    # Should have 3 content blocks
    assert len(user_msg["content"]) == 3
    
    # Check base64 data URLs are properly formatted
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    
    assert user_msg["content"][2]["type"] == "image_url"
    assert user_msg["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_conversation_with_multiple_multi_image_turns():
    """Test a full conversation where multiple turns each have multiple images"""
    history = ConversationHistory(system_prompt="You are a helpful image analysis assistant.")
    
    # User turn 1: Multiple images
    user1 = (UserTurn()
             .add_text("What's the difference between these flowers?")
             .add_image(url="https://example.com/rose.jpg")
             .add_image(url="https://example.com/tulip.jpg"))
    history.add_user_turn(user1)
    
    # Assistant response
    assistant1 = AssistantTurn().add_text("The first is a rose with layered petals, the second is a tulip with a cup shape.")
    history.add_assistant_turn(assistant1)
    
    # User turn 2: More images
    user2 = (UserTurn()
             .add_text("Now compare these landscapes:")
             .add_image(url="https://example.com/mountain.jpg")
             .add_image(url="https://example.com/beach.jpg")
             .add_image(url="https://example.com/forest.jpg"))
    history.add_user_turn(user2)
    
    messages = history.to_litellm_format()
    
    # Should have: system + user1 + assistant1 + user2
    assert len(messages) == 4
    
    # Check user1 has 3 content blocks (1 text + 2 images)
    assert len(messages[1]["content"]) == 3
    
    # Check user2 has 4 content blocks (1 text + 3 images)
    assert len(messages[3]["content"]) == 4


# ============================================================================
# Integration Tests - Truncation + Multimodal
# ============================================================================

def test_truncate_multimodal_conversation():
    """Test truncation works correctly with multimodal content"""
    history = ConversationHistory(system_prompt="You are a vision assistant.")
    
    # Add several turns with images
    for i in range(5):
        user = (UserTurn()
                .add_text(f"Analyze image {i}")
                .add_image(url=f"https://example.com/image{i}.jpg"))
        assistant = AssistantTurn().add_text(f"Analysis of image {i}")
        history.add_user_turn(user).add_assistant_turn(assistant)
    
    # Truncate to last 2 turns
    messages = history.to_messages(n=2, truncate_strategy="from_start")
    
    # Should have system + 2 turns
    assert len(messages) == 3
    
    # Check that multimodal content is preserved
    assert len(messages[1]["content"]) == 2  # text + image
    assert messages[1]["content"][1]["type"] == "image_url"


if __name__ == "__main__":
    print("Running optimizer backbone tests...")
    print("\n" + "="*80)
    print("TRUNCATION TESTS")
    print("="*80)
    
    test_default_all_history()
    print("✓ Default all history")
    
    test_truncate_from_start()
    print("✓ Truncate from start")
    
    test_truncate_from_end()
    print("✓ Truncate from end")
    
    test_truncate_zero_turns()
    print("✓ Truncate zero turns")
    
    test_truncate_more_than_available()
    print("✓ Truncate more than available")
    
    test_empty_conversation()
    print("✓ Empty conversation")
    
    test_to_litellm_format_with_truncation()
    print("✓ LiteLLM format with truncation")
    
    test_invalid_strategy()
    print("✓ Invalid strategy error handling")
    
    test_negative_n_values()
    print("✓ Negative n values")
    
    print("\n" + "="*80)
    print("MULTIMODAL TESTS")
    print("="*80)
    
    test_user_turn_multiple_images()
    print("✓ User turn with multiple images")
    
    test_assistant_turn_multiple_images()
    print("✓ Assistant turn with multiple images")
    
    test_mixed_content_types_in_turn()
    print("✓ Mixed content types in turn")
    
    test_multiple_images_with_base64()
    print("✓ Multiple base64 images")
    
    test_conversation_with_multiple_multi_image_turns()
    print("✓ Conversation with multiple multi-image turns")
    
    print("\n" + "="*80)
    print("INTEGRATION TESTS")
    print("="*80)
    
    test_truncate_multimodal_conversation()
    print("✓ Truncate multimodal conversation")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

