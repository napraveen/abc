import GetUserDetails from '../functions/GetUserDetails';
import { Link } from 'react-router-dom';
import '../css/Home.css';
import bestProduct from '../images/products.jpg';
import Header from './Header';
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import AvailableProducts from './AvailableProducts';
const Home = () => {
  const { userDetails } = GetUserDetails();

  return (
    <>
      <div className="home-home_page">
        {userDetails ? (
          <>
            <AvailableProducts />
          </>
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
