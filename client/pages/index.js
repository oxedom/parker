import DefaultLayout from "../layouts/DefaultLayout";

const Home = () => {
  return (
    <DefaultLayout>
      <div className="">
        <main className="">
          <h2 className="text-3xl font-bold pt-5 bg-green-800">
            {" "}
            About parker...
          </h2>
          <div className="bg-green-500 p-1 rounded-md">
            <p>
              {" "}
              Hello Everyone and Welcome to parker.com, Parker is a first of
              it's kind Web app that uses tensorflowJS
            </p>
          </div>
        </main>
      </div>
    </DefaultLayout>
  );
};

export default Home;
