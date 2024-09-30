import Header from './Header';
import '../css/product.css';
import React, { useEffect, useState } from 'react';
import axios from 'axios';

import GetUserDetails from '../functions/GetUserDetails';
const Favourites = () => {
  const { userDetails } = GetUserDetails();

  const [products, setProducts] = useState([]);

  const serverOrigin = process.env.REACT_APP_SERVER_ORIGIN;

  useEffect(() => {
    const fetchproducts = async () => {
      try {
        const res = await axios.get(`${serverOrigin}/likedProducts`);
        setProducts(res.data);
      } catch (err) {
        console.log('Error fetching the products data:', err);
      }
    };

    fetchproducts();
  }, []);
  const [visibleProductIndex, setVisibleProductIndex] = useState(null);

  const handleIssueTo = (index) => {
    if (visibleProductIndex === index) {
      setVisibleProductIndex(null);
    } else {
      setVisibleProductIndex(index);
    }
  };
  const [username, setUsername] = useState('');

  const [filteredRes, setFilteredRes] = useState('');
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

      const FilteredRes = await axios.get(
        `${serverOrigin}/filteredRes/${prediction}`
      );
      setProducts(FilteredRes.data);
    } catch (error) {
      console.error('Error uploading the file', error);
    }
  };

  const numberOfStars = 4;

  console.log('products:', products);
  const handleLike = async (productId) => {
    try {
      const res = await axios.post(`${serverOrigin}/handleLike`, {
        userId: userDetails._id, // Assuming userDetails contains the user's info
        productId,
      });
      console.log(res.data.message);
    } catch (err) {
      console.error('Error liking the product:', err);
    }
  };

  return (
    <div>
      <Header />

      <div className="product-body">
        <form onSubmit={handleSubmit}>
          <label
            for="file-upload"
            class="custom-file-upload"
            style={{
              backgroundColor: '#a4ff9f',
              height: '100px',
              width: '200px',
              borderRadius: '6px',
              padding: '6px',
              cursor: 'pointer',
            }}
          >
            Choose a file
          </label>

          <input
            type="file"
            id="file-upload"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />

          <button
            type="submit"
            style={{ width: '50px', backgroundColor: 'white' }}
          >
            {' '}
            🔎
          </button>
        </form>
        <div className="all-products">
          {products.map((product, index) => (
            <div
              className="product-container"
              key={index}
              style={{ height: '320px', width: '250px' }}
            >
              <a
                href={product.fileUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                <img
                  src={product.fileUrl}
                  alt={product.productname}
                  className="product-logos"
                />
              </a>

              {Array(5)
                .fill(0)
                .map((_, index) => (
                  <i
                    key={index}
                    class="fa fa-star"
                    style={
                      index < numberOfStars
                        ? { color: '#E6EE00' }
                        : { color: '#CDCDCD' }
                    }
                  ></i>
                ))}
              <i
                class="fas fa-heart"
                style={{
                  fontSize: '20px',
                  color: '#CDCDCD',
                  marginLeft: '100px',
                  cursor: 'pointer',
                  color: product.likeCount > 0 ? 'red' : 'black',
                }}
                onClick={() => handleLike(product._id)}
              ></i>
              <h4>
                <a
                  href={product.fileUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {product.productname}
                </a>
              </h4>
              <p>$ {product.price}</p>
              {userDetails && userDetails.username === 'admin' ? (
                <></>
              ) : (
                <>
                  <p
                    className="product-description"
                    style={{
                      height: '90px',
                      overflow: 'hidden',
                      width: '250px',
                    }}
                  >
                    {product.description}
                  </p>
                </>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Favourites;
