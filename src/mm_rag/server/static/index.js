function scrollToBottom() {
  const chatContainer = document.querySelector(".chat-container");
  chatContainer.scrollTop = chatContainer.scrollHeight;
};


class Message {
constructor(role, content) {
  this.role = role;
  this.content = content;
  this.date = new Date(); 
}

display() {
  const newMessage = document.createElement("div");
  if (this.role === 'user') {
    newMessage.classList.add("userMessage");
    newMessage.innerText = this.content;
    document.querySelector(".message-container").appendChild(newMessage);
    newMessage.style.display = 'block';
  scrollToBottom()
  }
}

async sendFetchAndDisplay() {

  //Creating the new message and putting a place holder
  const newMessage = document.createElement('div');
  newMessage.classList.add("aiMessage");
  document.querySelector(".message-container").appendChild(newMessage);
  newMessage.style.display = 'block';

  //Placeholder
  newMessage.innerHTML = 'Generating...'
  
  const response = await fetch('/talk', {
      method: "POST",
      headers: {
          'Content-Type': "application/json"
      },
      body: JSON.stringify({
          role: this.role,
          content: this.content,
          date: this.date
      })
  });

  if (!response.ok) {
      if (response.status === 500) {
          newMessage.innerText = ''
          const retryBtn = document.createElement('button')
          retryBtn.classList.add('retry-button')
          retryBtn.textContent = 'Error generating the model answer, please try again.'
          document.querySelector(".message-container").appendChild(retryBtn);
          retryBtn.addEventListener('click', () => {
              this.sendFetchAndDisplay()
          })
      }
      return; // Exit if the response is not OK
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  newMessage.innerHTML = ''
  try {
      while (true) {
          const { done, value } = await reader.read();
          if (done) {
              break; // Exit the loop when the stream is done
          }
          const chunk = decoder.decode(value, { stream: true });
          try{
              newMessage.innerHTML += marked.parse(chunk)
          } catch (error) {
              console.error("Error parsing the chunk:", error);
          }
      }
  } catch (error) {
      console.error("Stream error:", error);
  } finally {
      reader.releaseLock();
  }
}
};


document.addEventListener("DOMContentLoaded", () => {
scrollToBottom(); // Initial scroll to bottom
});

const sendButton = document.getElementById("inputBtn");

sendButton.addEventListener('click', function() {
const messageInput = document.getElementById("userInput");

if (messageInput.value.trim().length > 0 && messageInput.value.trim() !== '\n' && messageInput.value.trim() !== '\n\n') {
  let userMessage = new Message('user', messageInput.value);
  userMessage.display();
  let response = userMessage.sendFetchAndDisplay();
}
messageInput.value = ""; // Clear input

})

//Send message with enter key or go to a new line if Shift + Enter
shift_pressed = false

const inputBar = document.getElementById('userInput');
inputBar.addEventListener('keydown', (e) => {
  if (e.key === 'Shift') {
      shift_pressed = true
      window.setTimeout(() => {
          shift_pressed = false
      }, '1000')
  }
  else if (e.key === 'Enter' && shift_pressed) {
      inputBar.value += '\n'
      e.preventDefault()
  }
  else if (e.key === 'Enter' && !shift_pressed) {
      sendButton.click();
      e.preventDefault();
  }
})