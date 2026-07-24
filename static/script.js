document.getElementById("advisor-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const userInput = {
    n: parseFloat(document.getElementById("n").value),
    p: parseFloat(document.getElementById("p").value),
    k: parseFloat(document.getElementById("k").value),
    temperature: parseFloat(document.getElementById("temperature").value),
    humidity: parseFloat(document.getElementById("humidity").value),
    ph: parseFloat(document.getElementById("ph").value),
    rainfall: parseFloat(document.getElementById("rainfall").value),
    area: parseFloat(document.getElementById("area").value)
  };

  const resultDiv = document.getElementById("result");

  resultDiv.innerHTML = "Processing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(userInput)
    });

    const data = await response.json();

    if (data.results) {
      resultDiv.innerHTML = "<h3>Recommendations</h3>";

      data.results.forEach((res, index) => {
        resultDiv.innerHTML += `
          <div style="margin-bottom:15px;">
            <strong>${index + 1}. ${res.crop}</strong><br>
            Confidence: ${res.confidence}<br>
            Yield/ha: ${res.yield_per_hectare}<br>
            Total Yield: ${res.total_yield}
          </div>
        `;
      });

    } else {
      resultDiv.innerHTML = "<p style='color:red;'>Error getting results</p>";
    }

  } catch (error) {
    console.error(error);
    resultDiv.innerHTML = "<p style='color:red;'> Server error</p>";
  }
});