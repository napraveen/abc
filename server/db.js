const mongoose = require('mongoose');
const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: [true, 'Your email address is required'],
    unique: true,
  },
  username: {
    type: String,
    required: [true, 'Your username is required'],
  },
  password: {
    type: String,
    required: [true, 'Your password is required'],
  },
  verified: {
    type: String,
    required: [false],
    default: '',
  },
  createdAt: {
    type: Date,
    default: new Date(),
  },
});

const productSchema = new mongoose.Schema({
  productname: String,
  price: String,
  description: String,
  fileUrl: String,
});

const IssuedProductSchema = new mongoose.Schema({
  productname: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Product',
    required: [true, 'Product reference is required'],
  },
  availedUser: {
    type: String,
    required: [false],
  },
  date: {
    type: Date,
    default: Date.now,
  },
});

const Product = mongoose.model('Product', productSchema);

const User = mongoose.model('User', userSchema);
const IssuedProduct = mongoose.model('IssuedProduct', IssuedProductSchema);
module.exports = { User, Product, IssuedProduct };
