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

// const productSchema = new mongoose.Schema({
//   productname: String,
//   price: String,
//   description: String,
//   fileUrl: String,
// });

const productSchema = new mongoose.Schema({
  productname: { type: String, required: true },
  price: { type: Number, required: true },
  description: { type: String, required: true },
  fileUrl: { type: String, required: true },
  category: { type: String, required: true },
  brand: { type: String },
  stock: { type: Number, required: true },

  ratings: { type: Number, default: 0 },
  reviews: [
    {
      userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
      comment: { type: String },
      rating: { type: Number },
    },
  ],

  colorVariants: [String],
  sizeVariants: [String],
  discount: { type: Number, default: 0 },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
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
