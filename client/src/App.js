import { Route, Routes } from 'react-router-dom';
import Login from './pages/Login';
import Home from './pages/Home';
import Signup from './pages/Signup';
import AvailableProducts from './pages/AvailableProducts';
import Upload from './pages/Upload';
import User from './pages/Users';
import PageNotFound from './pages/PageNotFound';
import Test from './pages/Test';
function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/availableproducts" element={<AvailableProducts />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/users" element={<User />} />
        <Route path="/test" element={<Test />} />
        <Route path="*" element={<PageNotFound />}></Route>
      </Routes>
    </div>
  );
}

export default App;
