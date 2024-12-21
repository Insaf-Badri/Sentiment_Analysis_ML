async function getPrediction() {
    const input = document.getElementById('input_data'); 
    const output = document.getElementById('output_data'); 
    const probabilityChart = document.getElementById('probabilityChart');


    
    const inputText = input.value.trim(); 
    output.textContent = '';
  
    if (!inputText) {
      output.textContent = 'Please enter some text!';
      return;
    }
  
    output.textContent = 'Analyzing emotion...';
  
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
  
      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }
  
      const data = await response.json();
      output.textContent = `Emotion : ${data.emotion}`;
      // Display the chart image
      document.getElementById('probabilityChart').src = "data:image/png;base64," + data.probabilities_chart;
        
    } catch (error) {
      
      output.textContent = 'Error: Unable to analyze text. Please try again.';
      console.error('Error:', error);
    }

  }
  