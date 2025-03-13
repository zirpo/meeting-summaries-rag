#!/usr/bin/env python3
"""
Simple script to create a new meeting summary file.
"""

import argparse
import datetime
import os
from pathlib import Path

def create_meeting_summary(
    topic: str,
    date: str = None,
    participants: str = None,
    output_dir: str = "./meetings",
):
    """
    Create a new meeting summary file with the correct format
    
    Args:
        topic: Meeting topic
        date: Meeting date (YYYY-MM-DD)
        participants: Comma-separated list of participants
        output_dir: Directory to save the file
    
    Returns:
        Path to the created file
    """
    # Set default date to today if not provided
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Validate date format
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up topic for filename
    filename_topic = topic.replace(" ", "_").replace("/", "-").replace("\\", "-")
    
    # Create filename
    filename = f"{date}_{filename_topic}.md"
    file_path = os.path.join(output_dir, filename)
    
    # Create file content
    content = f"""# {topic}

Date: {date}
"""
    
    if participants:
        content += f"Participants: {participants}\n"
    
    content += """
## Agenda
- 
- 
- 

## Discussion
- 

## Decisions
- 

## Action Items
- 

## Next Steps
- 

## Next Meeting
- 
"""
    
    # Write file
    with open(file_path, "w") as f:
        f.write(content)
    
    return file_path


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a new meeting summary file")
    parser.add_argument(
        "topic",
        type=str,
        help="Meeting topic",
    )
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        help="Meeting date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--participants",
        "-p",
        type=str,
        help="Comma-separated list of participants",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./meetings",
        help="Directory to save the file (default: ./meetings)",
    )
    args = parser.parse_args()
    
    # Create meeting summary
    try:
        file_path = create_meeting_summary(
            topic=args.topic,
            date=args.date,
            participants=args.participants,
            output_dir=args.output_dir,
        )
        print(f"Created meeting summary file: {file_path}")
        print("To index this file, run: python test_rag.py")
    except Exception as e:
        print(f"Error creating meeting summary file: {e}")