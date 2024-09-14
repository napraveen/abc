import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post(
        'http://localhost:8000/myapp/classify/',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setPrediction(response.data.predicted_class);
    } catch (error) {
      console.error('Error uploading the file', error);
    }
  };

  return (
    <div>
      <h1>Image Classification</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Upload and Classify</button>
      </form>
      {prediction && <h2>Prediction: {prediction}</h2>}
    </div>
  );
}

export default App;
