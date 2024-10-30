css = '''
<style>
.chat-message {
    padding: 1.25rem;
    border-radius: 0.75rem;
    margin-bottom: 1.25rem;
    display: flex
}
.chat-message.user {
    background-color: #3a4252
}
.chat-message.bot {
    background-color: #5a6477
}
.chat-message .avatar {
  width: 18%;
}
.chat-message .avatar img {
  max-width: 72px;
  max-height: 72px;
  border-radius: 40%;
  object-fit: cover;
}
.chat-message .message {
  width: 82%;
  padding: 0 1.25rem;
  color: #f0f0f0;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
       <p>NEXTORY RESPONSE</p>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <p>QUERY</p>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''