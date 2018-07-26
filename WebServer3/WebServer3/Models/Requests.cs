using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Web;

namespace WebServer3.Models
{
    public class Requests
    {
        public string Body { get; set; }
        public string Uri { get; set; }


        public string HttpReqs(Requests req)
        {
            string ret;

            using (var client = new HttpClient())
            {
                var response = client.PostAsync(req.Uri, new StringContent(req.Body, Encoding.UTF8, "application/json"));
                ret = response.Result.Content.ReadAsStringAsync().Result;                
            }

            return ret;
        }
    }
}