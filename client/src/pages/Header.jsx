import React from 'react';
import { Link } from 'react-router-dom';

import GetUserDetails from '../functions/GetUserDetails';
import ecomLogo from '../images/products.jpg';
const Header = () => {
  const { userDetails } = GetUserDetails();

  const handleLogout = () => {
    const serverOrigin = process.env.REACT_APP_SERVER_ORIGIN;

    fetch(`${serverOrigin}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.message === 'Logged out successfully') {
          window.location.href = '/';
        }
      })
      .catch((error) => {
        console.error('Error logging out:', error);
      });
  };
  return (
    <div>
      {userDetails ? (
        <header>
          {' '}
          <div className="header-left">
            {' '}
            <img
              src={ecomLogo}
              alt="Logo"
              className="my-logo"
              width="30px"
              height="30px"
            />
            &nbsp;
          </div>
          <div className="header-middle">
            <p>
              {' '}
              <Link
                to="/"
                style={{ textDecoration: 'none' }}
                className="home-header-text"
              >
                Home
              </Link>
            </p>
            <p>
              {' '}
              <Link
                to="/favourites"
                style={{ textDecoration: 'none' }}
                className="home-header-text"
              >
                Favourites
              </Link>
            </p>
            {userDetails.username === 'admin' ? (
              <>
                {' '}
                <p>
                  {' '}
                  <Link
                    to="/upload"
                    style={{ textDecoration: 'none' }}
                    className="home-header-text"
                  >
                    Upload
                  </Link>
                </p>
                <p>
                  {' '}
                  <Link
                    to="/users"
                    style={{ textDecoration: 'none' }}
                    className="home-header-text"
                  >
                    Users
                  </Link>
                </p>
              </>
            ) : (
              ''
            )}
          </div>
          <div className="header-right">
            <Link to="/">
              <button className="logout" onClick={handleLogout}>
                Logout
              </button>
            </Link>
          </div>
        </header>
      ) : (
        <header>
          {' '}
          <div className="header-left">
            {' '}
            <img
              src={ecomLogo}
              alt="Logo"
              className="my-logo"
              width="30px"
              height="30px"
            />
            &nbsp;
          </div>
          <div className="header-middle">
            <p>
              {' '}
              <Link
                to="/"
                style={{ textDecoration: 'none' }}
                className="home-header-text"
              >
                Home
              </Link>
            </p>
            <p>
              {' '}
              <Link
                to="/availableproducts"
                style={{ textDecoration: 'none' }}
                className="home-header-text"
              >
                Available Products
              </Link>
            </p>
          </div>
          <div className="header-right">
            <Link to="/login" style={{ textDecoration: 'none' }}>
              {' '}
              <button className="login">Login</button>
            </Link>
          </div>
        </header>
      )}
    </div>
  );
};

export default Header;
