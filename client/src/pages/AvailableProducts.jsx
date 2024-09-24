import Header from './Header';
import '../css/product.css';
import React, { useEffect, useState } from 'react';
import axios from 'axios';

import GetUserDetails from '../functions/GetUserDetails';
const AvailableProducts = () => {
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
  const [visibleProductIndex, setVisibleProductIndex] = useState(null);

  const handleIssueTo = (index) => {
    if (visibleProductIndex === index) {
      setVisibleProductIndex(null);
    } else {
      setVisibleProductIndex(index);
    }
  };
  const [username, setUsername] = useState('');

  const handleSubmit = async (productId) => {
    try {
      // const userRes = await axios.get(`${serverOrigin}/api/user/${username}`);
      // const userId = userRes.data._id;

      await axios.post(`${serverOrigin}/issue-product`, {
        productId,
        username,
      });

      alert('Product issued successfully!');
    } catch (err) {
      console.error('Error issuing Product:', err);
    }
  };

  console.log('products:', products);
  return (
    <div>
      <Header />
      <div className="product-body">
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

              <i class="fa fa-star" style={{ color: '#E6EE00' }}></i>

              <i
                class="fas fa-heart"
                style={{ fontSize: '25px', color: '#CDCDCD' }}
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

export default AvailableProducts;
