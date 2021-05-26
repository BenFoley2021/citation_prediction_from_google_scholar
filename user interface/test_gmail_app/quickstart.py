# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:45:56 2021
runs as of 


https://mailtrap.io/blog/send-emails-with-gmail-api/

body of message is base64 encoded https://developers.google.com/gmail/api/reference/rest/v1/users.messages.attachments#MessagePartBody


@author: bcyk5
"""

from __future__ import print_function
import os.path
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

import email
import email.mime.text as mail
import base64
import pickle
from bs4 import BeautifulSoup 
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)

    # Call the Gmail API
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        print('No labels found.')
    else:
        print('Labels:')
        for label in labels:
            print(label['name'])

    return service

def create_message(sender, to, subject, message_text):
  message = mail.MIMEText(message_text)
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject
  raw_message = base64.urlsafe_b64encode(message.as_string().encode("utf-8"))
  return {
    'raw': raw_message.decode("utf-8")
  }


def create_message2(sender, to, subject, message_text):
  """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.

  Returns:
    An object containing a base64url encoded email object.
  """
  message = mail.MIMEText(message_text)
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject
  return {'raw': base64.urlsafe_b64encode(message.as_string())}



def create_draft(service, user_id, message_body):
  try:
    message = {'message': message_body}
    draft = service.users().drafts().create(userId=user_id, body=message).execute()

    print("Draft id: %s\nDraft message: %s" % (draft['id'], draft['message']))

    return draft
  except Exception as e:
    print('An error occurred: %s' % e)
    return None  

def send_message(service, user_id, message):
  """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
  try:
    message = (service.users().messages().send(userId=user_id, body=message)
               .execute())
    #print 'Message Id: %s' % message['id']
    return message
  except Exception as e:# error:
    print('An error occurred: %s' % e)# % error)

def get_messages(service, user_id):
  try:
    return service.users().messages().list(userId=user_id).execute()
  except Exception as error:
    print('An error occurred: %s' % error)
    
def get_message(service, user_id, msg_id, format_type):
    # 'metadata' or 'full'
  try:
    return service.users().messages().get(userId=user_id, id=msg_id, \
                                          format=format_type).execute()
  except Exception as error:
    print('An error occurred: %s' % error)


def get_mime_message(service, user_id, msg_id):
  try:
    message = service.users().messages().get(userId=user_id, id=msg_id,
                                             format='raw').execute()
    print('Message snippet: %s' % message['snippet'])
    msg_str = base64.urlsafe_b64decode(message['raw'].encode("utf-8")).decode("utf-8")
    mime_msg = email.message_from_string(msg_str)

    return mime_msg
  except Exception as error:
    print('An error occurred: %s' % error)
    
def check_if_gs_update(message_list):
    # ['Subject'] = 'Perovskite solar - new results'
    email_ids = []
    for email_message in message_list:
        for item in email_message['payload']['headers']:
            if item['name'] == 'Subject' and item['value'] == 'Perovskite solar - new results':
                email_ids.append(email_message['id'])
            
    return email_ids
    
def get_emails_by_id(service, email_ids):
    update_list = []
    for email_id in email_ids:
        update_list.append(base64.urlsafe_b64decode(
            get_message(service, "me", email_id, 'full')['payload']["body"]['data']
            ))
        
    return update_list

def get_emails_by_id2(service, email_ids):
    update_list = []
    for email_id in email_ids:
        temp_email = get_message(service, "me", email_id, 'full')
        decoded = base64.urlsafe_b64decode(temp_email['payload']["body"]['data'])

        update_list.append(decoded)
        
    return update_list


def save_soups(update_list):
    # saves each html doc in the list as an html
    name_counter = 0
    file_path = os.getcwd() +"//updates_html//"
    
    for update in update_list:
        name_counter += 1
        thing = str(BeautifulSoup(update, 'html.parser'))
        pickle.dump(thing, open(file_path + "update " + str(name_counter), 'wb'))
        
        
if __name__ == '__main__':
    service = main()
        
    # message = create_message('cakeeater720@gmail.com', 'cakeeater720@gmail.com', \
    #                          'app test 5-11 1pm', 'message text')
            
    # send_message(service, "me", message)
        
    messages = get_messages(service, "me") ### gets first 100 message ids
        
    message_list = []
    for thing in messages['messages']: # get the metadata of messages
        message_list.append(get_message(service, "me", thing['id'], 'metadata'))
        
    update_ids = check_if_gs_update(message_list)
        
    update_list = get_emails_by_id2(service, update_ids)
    
    save_soups(update_list)
    #test_message = get_message(service, "me", test_id)
    # def message_test_1():
    
    #     gs_email = get_message(service, "me", message_id)
    #     raw = base64.urlsafe_b64decode(gs_email['payload']["body"]['data'])
    
    