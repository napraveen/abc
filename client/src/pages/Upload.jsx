import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../css/upload.css';
import Header from './Header';

import GetUserDetails from '../functions/GetUserDetails';
import PageNotFound from './PageNotFound';
const ImageUpload = () => {
  const { userDetails } = GetUserDetails();

  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [productData, setProductData] = useState({
    productname: '',
    price: '',
    description: '',
  });
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);

  const serverOrigin = process.env.REACT_APP_SERVER_ORIGIN;

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleChange = (event) => {
    const { name, value } = event.target;
    setProductData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleUpload = async (event) => {
    event.preventDefault();

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('productname', productData.productname);
      formData.append('price', productData.price);
      formData.append('description', productData.description);

      const response = await axios.post(
        `${serverOrigin}/api/upload-image`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      console.log('Image uploaded successfully:', response.data.imageUrl);
      setImageUrl(response.data.imageUrl);
      setProductData({ productname: '', price: '', description: '' });
      setSelectedFile(null);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  useEffect(() => {
    const fetchproducts = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${serverOrigin}/products`);
        setProducts(response.data);
      } catch (error) {
        console.error('Error fetching products:', error);
      }
      setLoading(false);
    };

    fetchproducts();
  }, []);

  return (
    <div>
      {userDetails && userDetails.username === 'admin' ? (
        <>
          {' '}
          <Header />
          <div className="upload-container">
            {' '}
            <h2>Image Upload</h2>
            <form onSubmit={handleUpload}>
              <input type="file" name="file" onChange={handleFileChange} />
              <input
                type="text"
                name="productname"
                value={productData.productname}
                onChange={handleChange}
                placeholder="Product Name"
              />
              <input
                type="text"
                name="price"
                value={productData.price}
                onChange={handleChange}
                placeholder="price"
              />
              <input
                type="text"
                name="description"
                value={productData.description}
                onChange={handleChange}
                placeholder="Description"
              />
              <button type="submit">Upload</button>
            </form>
          </div>
        </>
      ) : (
        <PageNotFound />
      )}
    </div>
  );
};

export default ImageUpload;
