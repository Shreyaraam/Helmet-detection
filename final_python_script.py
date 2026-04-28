<!DOCTYPE html>
<html>
<head>
    <title>My Store</title>
</head>
<body>

    <h1>My Uncle's Store</h1>

    <h2>Add Product</h2>

    Name: <input id="name"><br><br>
    Price: <input id="price" type="number"><br><br>
    GST %: <input id="gst" type="number"><br><br>
    Quantity: <input id="qty" type="number"><br><br>

    <button onclick="addProduct()">Add Product</button>

    <h2>Products List</h2>
    <ul id="productList"></ul>

<script>

let products = JSON.parse(localStorage.getItem("products")) || [];

displayProducts();

function addProduct() {
    let name = document.getElementById("name").value;
    let price = document.getElementById("price").value;
    let gst = document.getElementById("gst").value;
    let qty = document.getElementById("qty").value;

    let product = {
        name: name,
        price: Number(price),
        gst: Number(gst),
        quantity: Number(qty)
    };

    products.push(product);

    saveToStorage();
    displayProducts();
}

function saveToStorage() {
    localStorage.setItem("products", JSON.stringify(products));
}

function displayProducts() {
    let list = document.getElementById("productList");
    list.innerHTML = "";

    products.forEach((p, index) => {
        let item = document.createElement("li");
        item.innerText = p.name + " | ₹" + p.price + 
                         " | GST: " + p.gst + "%" + 
                         " | Qty: " + p.quantity;

        list.appendChild(item);
    });
}

</script>

</body>
</html>
