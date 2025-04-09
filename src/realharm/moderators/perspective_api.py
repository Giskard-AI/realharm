"""Perspective API"""
import os
import time
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator


class PerspectiveModerator(BaseModerator):
    def __init__(self):
        self.threshold = 0.5    
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey= os.getenv("PERSPECTIVE_API_KEY"),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
            )


    def check(self, conversation: list[dict[str, str]], is_sample_safe: bool, **kwargs):
        
        for i, msg in enumerate(conversation):
            if is_sample_safe and msg["role"] == "user":
                continue
            time.sleep(1) # Perspective API has a rate limit of 60 request per minute
            analyze_request = {
                'comment': { 'text': msg["content"] },
                'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}}
                }
            
            try:
                response = self.client.comments().analyze(body=analyze_request).execute()
            except HttpError as e:
                # Perspective APOI does not support all languages, 
                # It raises an error if the language is not supported by the attribute
                # If the language is not support we consider the message as safe
                if e.error_details[0]["errorType"] == "LANGUAGE_NOT_SUPPORTED_BY_ATTRIBUTE":
                    print(e.error_details)
                    continue
                raise e
            
            attributes = list(response["attributeScores"].keys())
            is_safe = True
            categories = []
            for attribute in attributes:
                if response["attributeScores"][attribute]["summaryScore"]["value"] > self.threshold:
                    is_safe = False
                    categories.append(attribute)
                
            if not is_safe:
                return ModerationOutput(safe=False, categories=categories)

        return ModerationOutput(safe=True, categories=[])
