// import $ from "jquery"
var script = document.createElement('script');
script.src = 'https://code.jquery.com/jquery-3.4.1.min.js';
script.type = 'text/javascript';
document.getElementsByTagName('head')[0].appendChild(script);

const actualBtn = document.getElementById('actual-btn');
const doneBtn = document.getElementById('done-btn');
const fileChosen = document.getElementById('file-chosen');

actualBtn.addEventListener('change', function() {
  fileChosen.textContent = this.files[0].name;
  doneBtn.style.display = 'block';
});

doneBtn.addEventListener('click', function() {
  postData("./test_files/panda_grayscale.png")
});

function postData(input) {
  
  $.ajax({
      type: 'POST',
      url: '../../app.py',
      // data: { param: input },
      success: function(data) {
        alert(data['result']);
      }
  });
}

function callbackFunc(response) {
  // do something with the response
  console.log('done');
}