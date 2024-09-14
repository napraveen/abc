import GetUserDetails from '../functions/GetUserDetails';
import { Link } from 'react-router-dom';
import '../css/Home.css';
import bestBook from '../images/books.webp';
import Header from './Header';
import React, { useEffect, useState } from 'react';
import axios from 'axios';
const Home = () => {
  const { userDetails } = GetUserDetails();
  const [books, setBooks] = useState([]);
  const serverOrigin = process.env.REACT_APP_SERVER_ORIGIN;

  useEffect(() => {
    const fetchBooks = async () => {
      try {
        const res = await axios.get(`${serverOrigin}/books`);
        setBooks(res.data);
      } catch (err) {
        console.log('Error fetching the books data:', err);
      }
    };

    fetchBooks();
  }, []);
  const numberOfStars = 4;
  return (
    <>
      <div className="home-home_page">
        {userDetails ? (
          <div>
            <Header />
            <div className="book-body">
              {/* <h3>
        {' '}
        Hey there, Welcome to our
        <span> books collection!! </span>
      </h3> */}
              <div className="all-books">
                {books.map((book, index) => (
                  <div
                    className="book-container"
                    key={index}
                    style={{ height: '320px', width: '250px' }}
                  >
                    <a
                      href={book.fileUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <img
                        src={book.fileUrl}
                        alt={book.bookname}
                        className="book-logos"
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
                      style={{ fontSize: '20px', color: '#CDCDCD' , marginLeft:"100px"}}
                    ></i>
                    <h4>
                      <a
                        href={book.fileUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {book.bookname}
                      </a>
                    </h4>
                    <p>{book.author}</p>
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
                    visibleBookIndex === index ? 'visible' : 'hidden'
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
                    onClick={() => handleSubmit(book._id)}
                  />
                </div> */}
                      </>
                    ) : (
                      <>
                        <p
                          className="book-description"
                          style={{
                            height: '90px',
                            overflow: 'hidden',
                            width: '250px',
                          }}
                        >
                          {book.description}
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
                    Discover the world of <br></br>
                    <span style={{ color: '#ff4f00' }}> knowledge </span>
                    with us
                  </h1>
                  <p>
                    We believe in the transformative power of books and the joy
                    of reading. Our intuitive library management system is here
                    to help you find, borrow, and enjoy the books you love. Join
                    us on a journey through endless stories and limitless
                    learning. Your next adventure awaits!
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
                    src={bestBook}
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
