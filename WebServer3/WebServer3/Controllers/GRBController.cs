using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Web.Http;
using WebServer3.Models;
using System.IO;
using System.Web.Configuration;
using System.Web;

namespace ProductsApp.Controllers
{
    public class BGRController : ApiController
    {
        
        [HttpPost]
        public IHttpActionResult GetAllProducts()
        {
            Requests req = new Requests();
            req.Uri = WebConfigurationManager.AppSettings["localHost"];
            string reqBody = Request.Content.ReadAsStringAsync().Result;

            req.Body = "\"" + reqBody + "\""; //Fix Json

            //Send body along to Trevor (localhost:8080)
            string response = req.HttpReqs(req);
            int intruder = 0;

            if (response == "\"False\"") //Format response from Trevor
            {
                intruder = 1;
            }

            //Get response from Trevor and send to Mary in Return
            req.Uri = HttpUtility.HtmlDecode(WebConfigurationManager.AppSettings["flow"]);
            req.Body = "{'isarmed': 1,'isintruder': " + intruder + ",'emailaddress': 'typollak@microsoft.com','location': 'First Floor'}"; //Mock response from python server. 'intruder' boolean is currently returned
            response = req.HttpReqs(req);

            return Ok(response);
        }

        [HttpPost]
        public IHttpActionResult GetProduct(int id)
        {
            Requests req = new Requests();

            req.Body = "{'isarmed': 1,'isintruder': 0,'emailaddress': 'typollak@microsoft.com','location': 'First Floor'}";
            req.Uri = HttpUtility.HtmlDecode(WebConfigurationManager.AppSettings["flow"]);
            //pass reqBody to localhost:8080 for Trevor
            //pass json to Mary
            string response = req.HttpReqs(req);

            return Ok("URI: " + req.Uri + " Body: " + req.Body); //How to return response with json body
        }
    }
}