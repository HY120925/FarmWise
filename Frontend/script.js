document.getElementById("advisor-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const soil = document.getElementById("soil").value;
  const region = document.getElementById("region").value;

  try {
    const response = await fetch("http://127.0.0.1:5000/api/advisor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ soil_type: soil, region: region }),
    });

    const data = await response.json();
    console.log("API Response:", data);

    const resultDiv = document.getElementById("result");

    if (data.advisor_report) {
      const report = data.advisor_report;
      const res = report["Resource Requirements"];

      resultDiv.innerHTML = `
        <h3>Advisor Report</h3>
        <p><strong>Recommended Crop:</strong> ${report["Recommended Crop"]}</p>
        <p><strong>Expected Yield:</strong> ${report["Expected Yield"]}</p>
        <h4>Resource Requirements</h4>
        <ul>
          <li><strong>Crop:</strong> ${res.Crop}</li>
          <li><strong>Fertilizer Used:</strong> ${res["Fertilizer Used"]}</li>
          <li><strong>Yield Tons:</strong> ${res["Yield Tons"]}</li>
          <li><strong>Water Usage:</strong> ${res["Water Usage"]}</li>
        </ul>
      `;
    } else {
      resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
    }
  } catch (error) {
    console.error("Error fetching advisor:", error);
    document.getElementById("result").innerHTML =
      `<p style="color:red;">⚠️ Failed to connect to API</p>`;
  }
});
