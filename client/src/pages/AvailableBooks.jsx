import Header from './Header';
import '../css/book.css';
import React, { useEffect, useState } from 'react';
import axios from 'axios';

import GetUserDetails from '../functions/GetUserDetails';
const AvailableBooks = () => {
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
  const [visibleBookIndex, setVisibleBookIndex] = useState(null);

  const handleIssueTo = (index) => {
    if (visibleBookIndex === index) {
      setVisibleBookIndex(null);
    } else {
      setVisibleBookIndex(index);
    }
  };
  const [username, setUsername] = useState('');

  const handleSubmit = async (bookId) => {
    try {
      // const userRes = await axios.get(`${serverOrigin}/api/user/${username}`);
      // const userId = userRes.data._id;

      await axios.post(`${serverOrigin}/issue-book`, {
        bookId,
        username,
      });

      alert('Book issued successfully!');
    } catch (err) {
      console.error('Error issuing book:', err);
    }
  };

  console.log('Books:', books);
  return (
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
              <a href={book.fileUrl} target="_blank" rel="noopener noreferrer">
                <img
                  src={book.fileUrl}
                  alt={book.bookname}
                  className="book-logos"
                />
              </a>

              <i class="fa fa-star" style={{ color: '#E6EE00' }}></i>

              <i
                class="fas fa-heart"
                style={{ fontSize: '25px', color: '#CDCDCD' }}
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
  );
};

export default AvailableBooks;
