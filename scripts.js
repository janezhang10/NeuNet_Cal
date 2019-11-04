function EasyPeasyParallax() {
	scrollPos = $(this).scrollTop();
	$('.p1').css({
		'background-position' : '50% ' + (-scrollPos/4)+"px"
	});
  $('.p2').css({
		'background-position' : '50% ' + (-scrollPos/8)+"px"
	});
  $('.p3').css({
		'background-position' : '70% ' + (-scrollPos/16)+"px"
	});
	$('.parallax-text').css({
		'margin-top': (scrollPos/2)+"px",
		'opacity': 1-(scrollPos/230)
	});
}

$(document).ready(function(){
	$(window).scroll(function() {
		EasyPeasyParallax();
	});
});

var clicked = 0;


document.getElementById('bb').addEventListener('click', isRecord)

navigator.mediaDevices.getUserMedia({ audio: true })

function isRecord() {
	//alert("NeuNet wants to access your microphone"); 

	var counter = 0;
	clicked++;

	if (clicked%2 == 1) {
		document.getElementById('bb').innerHTML = 'MICROPHONE NOT DETECTED';
		//implement speech to text Python code
//		document.getElementById('bb').innerHTML = 'Stop Recording';
	}
	else {
		document.getElementById('bb').innerHTML = 'Start Recording';
		//ends speech to text Python code
	}
	
	alert("MICROPHONE NOT DETECTED"); 
	document.getElementById('bb').innerHTML = "MICROPHONE NOT DETECTED"
	document.getElementById('bb').style.color="red";
	document.getElementById('bb').style.background = 'white'
	
}

	
