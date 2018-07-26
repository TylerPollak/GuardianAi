using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Web;
using WebServer3.Models;

namespace WebServer3.Controllers
{
    public class Helper
    {
        public string HttpReqs(Requests req, string blah)
        {
            string ret;

            using (var client = new HttpClient())
            {
                var response = client.PostAsync(req.Uri, new StringContent(req.Body, Encoding.UTF8, "application/json"));
                ret = response.Result.Content.ToString();
            }

            return ret;
        }
    }
}