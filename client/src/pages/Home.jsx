import GetUserDetails from '../functions/GetUserDetails';
import { Link } from 'react-router-dom';
import '../css/Home.css';
import bestProduct from '../images/products.jpg';
import Header from './Header';
import React, { useEffect, useState } from 'react';
import axios from 'axios';
const Home = () => {
  const { userDetails } = GetUserDetails();
  const [products, setProducts] = useState([]);
  const serverOrigin = process.env.REACT_APP_SERVER_ORIGIN;

  useEffect(() => {
    const fetchproducts = async () => {
      try {
        const res = await axios.get(`${serverOrigin}/products`);
        setProducts(res.data);
      } catch (err) {
        console.log('Error fetching the products data:', err);
      }
    };

    fetchproducts();
  }, []);
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
  const numberOfStars = 4;
  return (
    <>
      <div className="home-home_page">
        {userDetails ? (
          <div>
            <Header />

            <div className="product-body">
              {/* <h3>
        {' '}
        Hey there, Welcome to our
        <span> products collection!! </span>
      </h3> */}
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
                {prediction && <h2>Prediction: {prediction}</h2>}
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
                      }}
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
                    <p>{product.price}</p>
                    {userDetails && userDetails.username === 'admin' ? (
                      <>
                        {' '}
                        {/* <button
                  onClick={() => handleIssueTo(index)}
                  className="issue-to-button"
                >
                  Issue to
                </button> */}
                        {/* <div
                  className={
                    visibleproductIndex === index ? 'visible' : 'hidden'
                  }
                >
                  <input
                    type="text"
                    placeholder="Enter Username of User"
                    className="issue-to-username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                  />
                  <input
                    type="button"
                    value="Submit"
                    className="issue-to-submit"
                    onClick={() => handleSubmit(product._id)}
                  />
                </div> */}
                      </>
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
                        {/* <i
                  class="fa fa-heart"
                  style={{ fontSize: '48px', color: 'red' }}
                ></i> */}
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            <Header />
            <div className="home-body">
              <div className="body-left">
                <div className="body-left-inside">
                  {' '}
                  <h1>
                    Experience the <br></br>
                    <span style={{ color: '#ff4f00' }}>
                      {' '}
                      future of online shopping{' '}
                    </span>
                    with us
                  </h1>
                  <p>
                    Browse through our vast collection of products, explore
                    exciting offers, and enjoy fast delivery. Get ready to shop
                    like never before!, we make shopping easy and fun. With a
                    wide range of products, you'll always find something new to
                    love.
                  </p>
                  <Link to="/login" style={{ textDecoration: 'none' }}>
                    <button className="get-started">Get Started</button>
                  </Link>
                </div>
              </div>
              <div className="body-right">
                <div className="body-right-img-container">
                  {' '}
                  <img
                    src={bestProduct}
                    alt="dashboard"
                    className="dashboard-img"
                  />
                </div>
              </div>
              <div className="body-right"></div>
            </div>
          </>
        )}
      </div>
    </>
  );
};

export default Home;
